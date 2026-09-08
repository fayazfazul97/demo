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
from report import build_summary, explain_split, list_assumptions
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
    tk = tk.merge(ss.backlog[["request_id", "source_account", "summary", "date_received"]], on="request_id", how="left")
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
        api_key = st.text_input("API key", type="password",
                                help="Only needed to run the AI pass. A cached proposal works without it.")
    model = ai_pass.DEFAULT_MODEL
    st.caption(f"Model: {model}")

config = merged_config(ss.config)   # refreshed again after account weights render

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Bet Scorer")
st.markdown(
    "Scores an inbound backlog and recommends a reactive / proactive capacity split. "
    "**The AI model reads and proposes tags. You review and edit. Code does the arithmetic.** "
    "Every AI proposal you change is logged."
)

# --------------------------------------------------------------------------- #
# How to use
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("""
**How to use this tool**

1. **Ingest.** Upload the backlog CSV (columns: request_id, date_received, source_account, request_type, summary, raw_notes, rough_effort_hint) or load the bundled Trellis backlog.
2. **AI pass.** Optionally edit the prompt, then run it. The AI model reads every record together and proposes root-cause clusters, severity, confidence, reactive/proactive classification, what each proactive item would retire, and flags contradictions. It does not score.
3. **Review.** Read every proposed tag. Change anything you disagree with and add a reviewer note. Add clusters the model missed. Adjust weights and thresholds in the sidebar and watch the results update live.
4. **Finalise.** Lock the tags and download the scored CSV, the summary with the split explanation and assumptions, the diff of everything you changed, and the finalised tags and config as JSON.

**Review the data before you trust the output.** Every tag the AI model proposes is a reading of free text and can be wrong: a cluster may group tickets that are different bugs, a severity may follow the tone of an email instead of the numbers, a tier may be misread, and a proactive item may be credited with retiring tickets it would only detect. The recommended split is arithmetic on those tags and on the sidebar config. It is only as defensible as the review you do in step 3. Do not present the output without reading the diff log and the assumptions tab.
""")

# --------------------------------------------------------------------------- #
# How scoring works
# --------------------------------------------------------------------------- #
with st.expander("How scoring works and what you can change", expanded=False):
    c = config
    st.markdown(f"""
**The formula**

Reactive tickets are scored per root-cause cluster, not one at a time. A ticket with no cluster is a cluster of one.

```
cluster_impact  = sum over member tickets of (account_weight x severity)
cluster_score   = cluster_impact x confidence / effort_points
```

Every member ticket inherits the cluster score and carries an equal share of the cluster's cost (`cost_share = effort_points / members`). This is the point of clustering: one fix, paid once, credited to every ticket it closes.

Proactive items are scored on their own impact plus the reactive impact they structurally retire:

```
impact          = account_weight[Internal] x severity
                  + sum of (account_weight x severity) over each ticket in `retires`
impact          = impact x metric_bonus        (only if metric_linked)
score           = impact x confidence / effort_points
```

A proactive item that retires four tickets competes on the same scale as those four tickets combined. That is how a structural fix can outrank a single loud request.

**Buckets, checked in this order**

1. `Redirect`: `redirect` is ticked. Not product work (config, data cleanup, process). Excluded from capacity.
2. `Discovery spike`: effort is `unclear` AND confidence is below **{c["discovery_confidence_max"]}** AND impact is at least **{c["discovery_min_impact"]}**. Costs the `unclear` effort points. Low-impact unknowns skip this and get scored on what is known.
3. `Do now`: score >= **{c["do_now_threshold"]}**.
4. `Defer`: score >= **{round(c["do_now_threshold"] * c["defer_ratio"], 2)}** (threshold x defer band).
5. `Decline`: everything else.

Then a capacity pass: Do-now items are taken in score order, each cluster's cost paid once, until **{c["quarter_capacity_points"]}** points are used. Anything that does not fit drops to Defer with an "over capacity" note.

**The split** is the share of committed points (Do now + Discovery spike) that is reactive vs proactive. Redirected work is left out because it does not consume roadmap capacity.

**Every parameter, and what moving it does**

| Parameter | Where it is set | Current | Effect |
|---|---|---|---|
| Account weight | Sidebar, per account in the file | {", ".join(f"{k} {v:g}" for k, v in sorted(c["account_weight"].items()))} | Multiplies every ticket's impact for that account. Seeded from the tier the AI reads in the notes (top/enterprise 5, mid 3, small 1, unknown 3). This is the assumption most worth arguing about, since GMV is never in the data. |
| Severity scale | Sidebar | low {c["severity_scale"]["low"]:g}, medium {c["severity_scale"]["medium"]:g}, high {c["severity_scale"]["high"]:g} | How much more a high-severity ticket counts than a low one. Widening the gap makes the ranking more about severity and less about account size. |
| Effort points | Sidebar | small {c["effort_points"]["small"]:g}, 1-2 sprints {c["effort_points"]["1-2 sprints"]:g}, large {c["effort_points"]["large"]:g}, unclear {c["effort_points"]["unclear"]:g} | The denominator. Raising `large` makes multi-quarter proactive bets harder to fund. `unclear` is the price of a discovery spike. |
| Metric-linked bonus | Sidebar | {c["metric_bonus"]:g} | Multiplier for proactive items whose notes name a company metric. Set to 1.0 to remove the advantage. |
| Do-now threshold | Sidebar | {c["do_now_threshold"]:g} | The score cut-off for funding. Lower it and the proactive share usually rises, because large structural items sit just under the line. |
| Defer band | Sidebar | {c["defer_ratio"]:g} | Fraction of the threshold that still earns Defer instead of Decline. |
| Discovery: confidence below | Sidebar | {c["discovery_confidence_max"]:g} | Unknown-effort items with confidence under this get a spike instead of a guessed score. |
| Discovery: impact at least | Sidebar | {c["discovery_min_impact"]:g} | Floor so a spike is not spent on an unknown at a tiny account. |
| Quarter capacity | Sidebar | {c["quarter_capacity_points"]:g} | Points the team can commit. Default assumes 5 people, 6 two-week sprints, about 5 points per sprint after support load. |
| Per-ticket tags | Review table | | `classification`, `cluster`, `severity`, `confidence`, `effort`, `redirect`, `metric`, `retires`. The AI proposes, you decide. Every change is logged in the diff. |
| Per-cluster tags | Review table | | `effort` and `confidence` for the shared fix. These override the member tickets' own values. |
| Custom clusters | Review, "Add a cluster" | | Create a cluster the AI missed, then assign tickets to it in the ticket table. Membership is defined by the ticket rows, not the cluster row. |
""")

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
elif use_bundled:
    if ss.get("backlog_name") != DATA_PATH.name:
        ss.ai_blob = None
        ss.finalized = False
    ss.backlog = read_backlog(DATA_PATH)
    ss.backlog_name = DATA_PATH.name


def empty_sections(from_step: int, why: str) -> None:
    """Render the remaining section headers with nothing in them."""
    titles = {2: "2. AI pass", 3: "3. Review and finalise", 4: "4. Results"}
    for n in range(from_step, 5):
        st.header(titles[n])
        st.caption(why)


if ss.backlog is None:
    st.info("Upload a CSV, or load the bundled Trellis backlog, to start. Nothing is loaded yet.")
    with st.expander("Raw backlog", expanded=False):
        st.caption("Empty until a file is loaded.")
    empty_sections(2, "Waiting for a backlog.")
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
    "One call with the whole backlog. The AI model proposes clusters, severity, confidence, classification, "
    "what each proactive item would retire, and flags contradictions. It is told not to score."
)
ss.setdefault("system_prompt", ai_pass.SYSTEM_PROMPT)
with st.expander("Prompt (editable before running)", expanded=False):
    st.caption("This is the instruction the AI model receives with the backlog. Edit it if you want different definitions or emphasis. "
               "The output structure (the JSON fields) is fixed by the schema and cannot be changed here.")
    edited_prompt = st.text_area("System prompt", value=ss.system_prompt, height=420, key="prompt_editor", label_visibility="collapsed")
    pc1, pc2, pc3 = st.columns([1, 1, 3])
    if pc1.button("Use this prompt", disabled=edited_prompt == ss.system_prompt):
        ss.system_prompt = edited_prompt
        st.rerun()
    if pc2.button("Reset to default", disabled=ss.system_prompt == ai_pass.SYSTEM_PROMPT):
        ss.system_prompt = ai_pass.SYSTEM_PROMPT
        ss.pop("prompt_editor", None)
        st.rerun()
    if ss.system_prompt != ai_pass.SYSTEM_PROMPT:
        pc3.caption("Custom prompt in use. A run with it is cached separately from the default.")
    with st.popover("Output schema"):
        st.json(ai_pass.TOOL_SCHEMA["input_schema"], expanded=False)

cached = ai_pass.load_cache(bl, model, ss.system_prompt)
cur_hash = ai_pass.prompt_hash(bl, model, ss.system_prompt)
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
    st.caption(f"Model: {model}" + ("  |  custom prompt" if ss.system_prompt != ai_pass.SYSTEM_PROMPT else ""))

if run_api:
    with st.spinner("Reading the backlog..."):
        try:
            blob = ai_pass.run_ai_pass(bl, api_key, model, ss.system_prompt)
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
    empty_sections(3, "Waiting for an AI proposal.")
    st.stop()

meta = ss.ai_blob
if meta.get("source") == "seed":
    st.warning("This is the seed proposal that ships with the repo, drafted offline, not a live model run. Run the AI pass so the tags come from the model.")
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

st.subheader("Clusters")
st.caption("Cluster effort and confidence override the member tickets' values. Every ticket in a cluster shares one fix. "
           "Membership is set per ticket in the table below; the `tickets` column here is what the AI proposed, for reference.")

with st.expander("Add a cluster the AI missed", expanded=False):
    a1, a2 = st.columns([1, 2])
    new_id = a1.text_input("cluster_id (snake_case)", key="new_cl_id", disabled=locked)
    new_label = a2.text_input("label", key="new_cl_label", disabled=locked)
    a3, a4 = st.columns(2)
    new_effort = a3.selectbox("effort", EFFORT_BUCKETS, index=1, key="new_cl_effort", disabled=locked)
    new_conf = a4.slider("confidence", 0.0, 1.0, 0.5, 0.05, key="new_cl_conf", disabled=locked)
    new_hyp = st.text_input("root cause hypothesis", key="new_cl_hyp", disabled=locked)
    if st.button("Add cluster", disabled=locked or not new_id.strip()):
        cid = new_id.strip().lower().replace(" ", "_")
        existing = ss.clusters["cluster_id"].astype(str).tolist() if len(ss.clusters) else []
        if cid in existing:
            st.error(f"'{cid}' already exists.")
        else:
            row = pd.DataFrame([{
                "cluster_id": cid, "label": new_label.strip() or cid, "ticket_ids": "",
                "root_cause_hypothesis": new_hyp.strip(), "evidence": "reviewer-created",
                "effort_bucket": new_effort, "confidence": float(new_conf),
            }])
            # keep any pending edits in the cluster editor, then append
            base = ss.get("_pending_clusters", ss.clusters)
            ss.clusters = pd.concat([base, row], ignore_index=True)
            ss.pop("cluster_editor", None)
            st.rerun()

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
ss._pending_clusters = working_clusters
cluster_ids = [""] + sorted(working_clusters["cluster_id"].astype(str).tolist()) if len(working_clusters) else [""]

if not locked and len(working_clusters):
    rm1, rm2 = st.columns([1, 3])
    victim = rm1.selectbox("Remove a cluster", [""] + [c for c in cluster_ids if c], key="rm_cl")
    if rm2.button("Remove", disabled=not victim):
        ss.clusters = working_clusters[working_clusters["cluster_id"] != victim].reset_index(drop=True)
        base_t = ss.get("_pending_tickets", ss.tickets).copy()
        base_t.loc[base_t["cluster_id"] == victim, "cluster_id"] = ""
        ss.tickets = base_t
        ss.pop("cluster_editor", None); ss.pop("ticket_editor", None); ss.pop("rm_cl", None)
        st.rerun()

st.subheader("Tickets")
st.caption("Assign a ticket to a cluster by picking a cluster_id. For proactive items, list the reactive tickets they retire as comma-separated ids.")
tk_editor = st.data_editor(
    ss.tickets,
    key="ticket_editor",
    disabled=locked or ["request_id", "source_account", "summary", "date_received", "reason", "ambiguity_note"],
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
ss._pending_tickets = working_tickets

# Sanity notes on cluster membership
if len(working_clusters):
    counts = working_tickets["cluster_id"].fillna("").astype(str).value_counts()
    lonely = [c for c in working_clusters["cluster_id"].astype(str) if counts.get(c, 0) < 2]
    if lonely:
        st.caption(f"Clusters with fewer than 2 tickets (scored as singletons until you assign more): {', '.join(lonely)}")
unknown = sorted(set(working_tickets["cluster_id"].fillna("").astype(str)) - set(cluster_ids))
if unknown:
    st.warning(f"Tickets reference clusters that do not exist: {', '.join(unknown)}. They will score on their own values.")

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

tiers_for_assumptions = meta["result"].get("accounts", [])
st.subheader("Why this split")
st.markdown(explain_split(scored, clusters_scored, split, config, working_tickets, working_clusters))

tabs = st.tabs(BUCKET_ORDER + ["Clusters", "Assumptions", "Diff log", "All scored"])
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
    st.markdown(list_assumptions(scored, clusters_scored, config, tiers_for_assumptions, meta, diff))
with tabs[len(BUCKET_ORDER) + 2]:
    if diff.empty:
        st.write("No changes from the AI proposal yet.")
    else:
        st.dataframe(diff, width='stretch', hide_index=True)
with tabs[len(BUCKET_ORDER) + 3]:
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
