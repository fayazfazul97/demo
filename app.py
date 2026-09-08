"""
Bet Scorer: reactive vs proactive backlog triage.

Flow: ingest CSV -> AI pass proposes tags -> reviewer edits and finalises
-> deterministic scoring, capacity split, exports.

Run:  streamlit run app.py
"""

from __future__ import annotations

import copy
import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

import ai_pass
from report import build_summary
from scoring import (
    BUCKET_ORDER,
    CLASSIFICATIONS,
    DEFAULT_CONFIG,
    EFFORT_BUCKETS,
    SEVERITIES,
    TIER_WEIGHTS,
    capacity_split,
    diff_tags,
    merged_config,
    read_backlog,
    score_backlog,
)

DATA_PATH = Path(__file__).parent / "data" / "Inbound_Requests.csv"

st.set_page_config(page_title="Bet Scorer", layout="wide")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("backlog", None)
ss.setdefault("ai_blob", None)        # raw cached/API response
ss.setdefault("ai_tickets", None)     # AI proposal, never edited
ss.setdefault("ai_clusters", None)
ss.setdefault("tickets", None)        # reviewer's working copy
ss.setdefault("clusters", None)
ss.setdefault("finalized", False)
ss.setdefault("config", copy.deepcopy(DEFAULT_CONFIG))


def seed_account_weights(tiers: dict[str, str] | None = None) -> None:
    """Make sure every account in the loaded file has a weight; seed new ones from AI tier."""
    aw = ss.config["account_weight"]
    tiers = tiers or {}
    present = set(ss.backlog["source_account"]) | {"Internal"}
    for acct in present:
        if acct not in aw:
            aw[acct] = TIER_WEIGHTS.get(tiers.get(acct, "unknown"), TIER_WEIGHTS["unknown"])
            ss.pop(f"aw_{acct}", None)
    for acct in list(aw):          # drop accounts from a previous file
        if acct not in present:
            del aw[acct]
            ss.pop(f"aw_{acct}", None)


def load_proposal(blob: dict) -> None:
    tiers = {a["source_account"]: a["tier"] for a in blob["result"].get("accounts", [])}
    # Loading a proposal starts a fresh review: weights are re-seeded from the
    # tiers the model read. The reviewer overrides them in the sidebar after.
    for acct, tier in tiers.items():
        if acct in ss.backlog["source_account"].values:
            ss.config["account_weight"][acct] = TIER_WEIGHTS.get(tier, 3)
            ss.pop(f"aw_{acct}", None)
    seed_account_weights(tiers)
    ss.account_tiers = blob["result"].get("accounts", [])
    tk, cl = ai_pass.payload_to_frames(blob["result"])
    tk["retires"] = tk["retires"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))
    tk = tk.merge(ss.backlog[["request_id", "source_account", "summary"]], on="request_id", how="left")
    tk["reviewer_note"] = ""
    ss.ai_blob = blob
    ss.ai_tickets = tk.copy()
    ss.ai_clusters = cl.copy()
    ss.tickets = tk.copy()
    ss.clusters = cl.copy()
    ss.finalized = False


# --------------------------------------------------------------------------- #
# Sidebar: scoring config
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Scoring config")
    st.caption("Every number a stakeholder could argue with lives here. Change one and watch the split move.")
    cfg = ss.config

    account_box = st.container()   # filled after the backlog is loaded, so it lists real accounts
    with st.expander("Effort points", expanded=False):
        for eb in EFFORT_BUCKETS:
            cfg["effort_points"][eb] = st.number_input(eb, 0.5, 20.0, float(cfg["effort_points"][eb]), 0.5, key=f"ep_{eb}")
    with st.expander("Severity scale", expanded=False):
        for sv in SEVERITIES:
            cfg["severity_scale"][sv] = st.number_input(sv, 0.5, 10.0, float(cfg["severity_scale"][sv]), 0.5, key=f"sv_{sv}")

    cfg["metric_bonus"] = st.slider("Metric-linked bonus (proactive)", 1.0, 3.0, float(cfg["metric_bonus"]), 0.1)
    cfg["do_now_threshold"] = st.slider("Do-now threshold", 0.5, 10.0, float(cfg["do_now_threshold"]), 0.25)
    cfg["defer_ratio"] = st.slider("Defer band (x threshold)", 0.1, 1.0, float(cfg["defer_ratio"]), 0.05)
    cfg["discovery_confidence_max"] = st.slider("Discovery spike if confidence below", 0.0, 1.0, float(cfg["discovery_confidence_max"]), 0.05)
    cfg["discovery_min_impact"] = st.slider("...and impact at least", 0.0, 20.0, float(cfg["discovery_min_impact"]), 0.5)
    cfg["quarter_capacity_points"] = st.number_input("Quarter capacity (points)", 1, 200, int(cfg["quarter_capacity_points"]))

    if st.button("Reset config to defaults"):
        for k in list(ss.keys()):
            if k.startswith(("aw_", "ep_", "sv_")):
                del ss[k]
        ss.config = copy.deepcopy(DEFAULT_CONFIG)
        st.rerun()

    st.divider()
    st.header("AI pass")
    # Key resolution: st.secrets (hosted) -> env var (local) -> sidebar input (fallback).
    server_key = ""
    try:
        server_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    server_key = server_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if server_key:
        api_key = server_key
        st.caption("API key is configured on the server.")
    else:
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Only needed to run the AI pass. A cached proposal works without it.")
    model = ai_pass.DEFAULT_MODEL
    st.caption(f"Model: {model}")

config = merged_config(ss.config)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Bet Scorer")
st.markdown(
    "Scores an inbound backlog and recommends a reactive / proactive capacity split. "
    "**Claude reads and proposes tags. You review and edit. Code does the arithmetic.** "
    "Every AI proposal you change is logged."
)

# --------------------------------------------------------------------------- #
# 1. Ingest
# --------------------------------------------------------------------------- #
st.header("1. Ingest")
c1, c2 = st.columns([2, 1])
with c1:
    uploaded = st.file_uploader("Backlog CSV (request_id, date_received, source_account, request_type, summary, raw_notes, rough_effort_hint)", type=["csv"])
with c2:
    st.write("")
    st.write("")
    use_bundled = st.button("Use bundled Trellis backlog", disabled=not DATA_PATH.exists())

if uploaded is not None and ss.get("backlog_name") != uploaded.name:
    ss.backlog = read_backlog(uploaded)
    ss.backlog_name = uploaded.name
    ss.ai_blob = None
    ss.finalized = False
elif use_bundled or (ss.backlog is None and DATA_PATH.exists()):
    if ss.get("backlog_name") != DATA_PATH.name:
        ss.ai_blob = None
        ss.finalized = False
    ss.backlog = read_backlog(DATA_PATH)
    ss.backlog_name = DATA_PATH.name

if ss.backlog is None:
    st.info("Upload a CSV to start.")
    st.stop()

bl = ss.backlog
seed_account_weights()
with account_box:
    with st.expander("Account weights", expanded=False):
        st.caption("Seeded from the tier the AI pass reads in the notes (top/enterprise 5, mid 3, small 1, unknown 3). Override freely.")
        cfg = ss.config
        for acct in sorted(cfg["account_weight"].keys()):
            cfg["account_weight"][acct] = st.number_input(acct, 0.0, 10.0, float(cfg["account_weight"][acct]), 0.5, key=f"aw_{acct}")
config = merged_config(ss.config)
st.caption(f"{ss.backlog_name}: {len(bl)} records, {bl['source_account'].nunique()} sources. "
           f"Effort hints normalised to: {dict(bl['effort_bucket'].value_counts())}")
with st.expander("Raw backlog", expanded=False):
    st.dataframe(bl.drop(columns=["effort_bucket"]), width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
# 2. AI pass
# --------------------------------------------------------------------------- #
st.header("2. AI pass")
st.markdown(
    "One call with the whole backlog. Claude proposes clusters, severity, confidence, classification, "
    "what each proactive item would retire, and flags contradictions. It is told not to score."
)
cached = ai_pass.load_cache(bl, model)
cur_hash = ai_pass.prompt_hash(bl, model)
MAX_RUNS_PER_SESSION = 3
ss.setdefault("ai_runs", 0)
runs_left = MAX_RUNS_PER_SESSION - ss.ai_runs

b1, b2, b3 = st.columns(3)
with b1:
    run_api = st.button(f"Run AI pass via API ({runs_left} left this session)", type="primary",
                        disabled=(not api_key) or runs_left <= 0)
with b2:
    load_cached = st.button("Load cached proposal", disabled=cached is None)
with b3:
    with st.popover("View prompt"):
        st.code(ai_pass.SYSTEM_PROMPT, language="markdown")
        st.json(ai_pass.TOOL_SCHEMA["input_schema"], expanded=False)

if run_api:
    with st.spinner("Reading the backlog..."):
        try:
            blob = ai_pass.run_ai_pass(bl, api_key, model)
            ss.ai_runs += 1
            ai_pass.save_cache(blob)
            load_proposal(blob)
            st.toast(f"AI pass complete and cached. Tokens: {blob['usage']}")
            st.rerun()   # re-render the sidebar with weights seeded from the new tiers
        except Exception as e:  # surface, don't hide
            st.error(f"AI pass failed: {e}")

if load_cached and cached:
    load_proposal(cached)
    st.rerun()

if ss.ai_blob is None:
    if cached is None:
        st.info("No cached proposal for this dataset yet. Run the AI pass to generate one.")
    else:
        st.caption(
            f"Cached proposal available: source={cached.get('source')}, model={cached.get('model')}, "
            f"generated={cached.get('generated_at')}. "
            + ("Prompt hash matches the current prompt and data." if cached.get("prompt_hash") == cur_hash
               else "Prompt hash differs from the current prompt/data/model; consider re-running.")
        )
    st.stop()

meta = ss.ai_blob
if meta.get("source") == "seed":
    st.warning("You are using the seed proposal that ships with the repo. Re-run the AI pass with an API key before submitting so the cache is a real model output.")
else:
    st.caption(f"Proposal from {meta.get('model')} at {meta.get('generated_at')} (source: {meta.get('source')}).")

tiers_list = meta["result"].get("accounts", [])
if tiers_list:
    with st.expander("Account tiers read from the notes", expanded=False):
        st.dataframe(pd.DataFrame(tiers_list), width='stretch', hide_index=True)

obs = meta["result"].get("cross_record_observations", [])
if obs:
    with st.expander("Cross-record observations", expanded=True):
        for o in obs:
            st.markdown(f"- {o}")

# --------------------------------------------------------------------------- #
# 3. Review
# --------------------------------------------------------------------------- #
st.header("3. Review and finalise")
locked = ss.finalized
if locked:
    st.success("Finalised. Editors are locked. Reopen to change anything.")
else:
    st.markdown("Edit anything you disagree with. Add a reviewer note so the change is explained in the diff log.")

cluster_ids = [""] + sorted(ss.clusters["cluster_id"].astype(str).tolist()) if len(ss.clusters) else [""]

st.subheader("Clusters")
st.caption("Cluster effort and confidence override the member tickets' values. Every ticket in a cluster shares one fix.")
cl_editor = st.data_editor(
    ss.clusters,
    key="cluster_editor",
    disabled=locked or ["cluster_id", "ticket_ids", "evidence", "root_cause_hypothesis"],
    hide_index=True,
    width='stretch',
    column_order=["cluster_id", "label", "ticket_ids", "effort_bucket", "confidence", "root_cause_hypothesis", "evidence"],
    column_config={
        "cluster_id": st.column_config.TextColumn("cluster_id", width="small"),
        "label": st.column_config.TextColumn("label", width="medium"),
        "ticket_ids": st.column_config.TextColumn("tickets", width="small"),
        "effort_bucket": st.column_config.SelectboxColumn("effort", options=EFFORT_BUCKETS, width="small"),
        "confidence": st.column_config.NumberColumn("confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
        "root_cause_hypothesis": st.column_config.TextColumn("root cause hypothesis", width="large"),
        "evidence": st.column_config.TextColumn("evidence", width="large"),
    },
)
working_clusters = ss.clusters if locked else cl_editor

st.subheader("Tickets")
st.caption("Assign a ticket to a cluster by picking a cluster_id. For proactive items, list the reactive tickets they retire as comma-separated ids.")
tk_editor = st.data_editor(
    ss.tickets,
    key="ticket_editor",
    disabled=locked or ["request_id", "source_account", "summary", "reason", "ambiguity_note"],
    hide_index=True,
    width='stretch',
    height=620,
    column_order=[
        "request_id", "source_account", "classification", "cluster_id", "severity", "confidence",
        "effort_bucket", "redirect", "metric_linked", "retires", "reviewer_note", "reason", "ambiguity_note", "summary",
    ],
    column_config={
        "request_id": st.column_config.TextColumn("id", width="small"),
        "source_account": st.column_config.TextColumn("account", width="small"),
        "classification": st.column_config.SelectboxColumn("class", options=CLASSIFICATIONS, width="small"),
        "cluster_id": st.column_config.SelectboxColumn("cluster", options=cluster_ids, width="small"),
        "severity": st.column_config.SelectboxColumn("severity", options=SEVERITIES, width="small"),
        "confidence": st.column_config.NumberColumn("conf", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
        "effort_bucket": st.column_config.SelectboxColumn("effort", options=EFFORT_BUCKETS, width="small"),
        "redirect": st.column_config.CheckboxColumn("redirect", width="small"),
        "metric_linked": st.column_config.CheckboxColumn("metric", width="small"),
        "retires": st.column_config.TextColumn("retires", width="medium"),
        "reviewer_note": st.column_config.TextColumn("reviewer note", width="medium"),
        "reason": st.column_config.TextColumn("AI reason", width="large"),
        "ambiguity_note": st.column_config.TextColumn("AI ambiguity note", width="large"),
        "summary": st.column_config.TextColumn("summary", width="large"),
    },
)
working_tickets = ss.tickets if locked else tk_editor

f1, f2, f3 = st.columns([1, 1, 4])


def _clear_editors() -> None:
    for k in ("ticket_editor", "cluster_editor"):
        ss.pop(k, None)


with f1:
    if st.button("Finalise", type="primary", disabled=locked):
        ss.tickets = working_tickets.copy()
        ss.clusters = working_clusters.copy()
        _clear_editors()
        ss.finalized = True
        st.rerun()
with f2:
    if st.button("Reopen review", disabled=not locked):
        ss.finalized = False
        st.rerun()
with f3:
    if st.button("Discard my edits, back to AI proposal", disabled=locked):
        ss.tickets = ss.ai_tickets.copy()
        ss.clusters = ss.ai_clusters.copy()
        _clear_editors()
        st.rerun()

# --------------------------------------------------------------------------- #
# 4. Results (live while reviewing, exports once finalised)
# --------------------------------------------------------------------------- #
st.header("4. Results" + ("" if locked else "  (live preview, finalise to export)"))

try:
    scored, clusters_scored = score_backlog(working_tickets, working_clusters, config)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

split = capacity_split(scored, config)
diff = diff_tags(ss.ai_tickets, working_tickets)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Reactive", f"{split['reactive_pct']}%", f"{split['reactive_points']} pts")
m2.metric("Proactive", f"{split['proactive_pct']}%", f"{split['proactive_points']} pts")
m3.metric("Committed", f"{split['committed_points']} / {split['capacity_points']}", f"{split['headroom_points']} headroom")
m4.metric("Backlog reactive by count", f"{split['reactive_in_backlog_pct']}%")
m5.metric("Fields changed vs AI", len(diff), f"{diff['request_id'].nunique()} tickets" if len(diff) else None)

st.caption("Split counts Do-now and Discovery-spike points. Redirected work is not roadmap capacity and is excluded. "
           "Clustered tickets each carry an equal share of one fix's cost.")

tabs = st.tabs(BUCKET_ORDER + ["Clusters", "Diff log", "All scored"])
for tab, bucket in zip(tabs[: len(BUCKET_ORDER)], BUCKET_ORDER):
    with tab:
        sub = scored[scored["bucket"] == bucket]
        if sub.empty:
            st.write("Nothing here.")
            continue
        st.dataframe(
            sub[["request_id", "source_account", "classification", "cluster_id", "severity", "eff_confidence",
                 "eff_effort_bucket", "impact", "cost_share", "score", "capacity_note", "reason"]],
            width='stretch', hide_index=True,
            column_config={"reason": st.column_config.TextColumn("reason", width="large")},
        )
with tabs[len(BUCKET_ORDER)]:
    st.dataframe(clusters_scored, width='stretch', hide_index=True)
with tabs[len(BUCKET_ORDER) + 1]:
    if diff.empty:
        st.write("No changes from the AI proposal yet.")
    else:
        st.dataframe(diff, width='stretch', hide_index=True)
with tabs[len(BUCKET_ORDER) + 2]:
    st.dataframe(scored.drop(columns=["summary"], errors="ignore"), width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
if locked:
    st.subheader("Exports")
    summary_md = build_summary(scored, clusters_scored, split, config, diff, meta, obs)
    final_json = {
        "config": config,
        "ai_pass": {k: meta.get(k) for k in ("source", "model", "generated_at", "prompt_hash")},
        "tickets": json.loads(working_tickets.to_json(orient="records")),
        "clusters": json.loads(working_clusters.to_json(orient="records")),
    }
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("Scored CSV", scored.to_csv(index=False), "scored_backlog.csv", "text/csv")
    e2.download_button("Summary (markdown)", summary_md, "summary.md", "text/markdown")
    e3.download_button("Diff log CSV", diff.to_csv(index=False), "ai_vs_reviewer_diff.csv", "text/csv")
    e4.download_button("Finalised tags + config JSON", json.dumps(final_json, indent=2), "finalized.json", "application/json")
    with st.expander("Preview summary.md"):
        st.markdown(summary_md)
