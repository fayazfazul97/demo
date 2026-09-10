"""
Backlog scorer: reactive vs proactive work.

Flow: load the backlog -> the AI model reads it and suggests tags ->
a person reviews and adjusts -> the maths runs -> download the results.

Run:  streamlit run app.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ai_pass
from report import build_summary, explain_split, gaps_table, list_assumptions
from scoring import (
    BUCKET_HELP,
    BUCKET_ORDER,
    CLASSIFICATIONS,
    DEFAULT_CONFIG,
    EFFORT_BUCKETS,
    FILL_OPTIONS,
    SEVERITIES,
    TIER_WEIGHTS,
    capacity_split,
    diff_tags,
    merged_config,
    read_backlog,
    score_backlog,
)

DATA_PATH = Path(__file__).parent / "data" / "Inbound_Requests.csv"
MAX_RUNS_PER_SESSION = 3
_cluster_graph = components.declare_component("cluster_graph", path=str(Path(__file__).parent / "graph_component"))

st.set_page_config(page_title="Backlog scorer", layout="wide")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("backlog", None)
ss.setdefault("backlog_name", None)
ss.setdefault("ai_blob", None)        # the saved model output, never edited
ss.setdefault("ai_tickets", None)     # model's ticket tags, never edited
ss.setdefault("ai_clusters", None)
ss.setdefault("accounts_df", None)    # model's account read + editable weight
ss.setdefault("tickets", None)        # reviewer's working copies
ss.setdefault("clusters", None)
ss.setdefault("finalized", False)
ss.setdefault("config", copy.deepcopy(DEFAULT_CONFIG))
ss.setdefault("ai_runs", 0)
ss.setdefault("system_prompt", ai_pass.SYSTEM_PROMPT)
ss.setdefault("graph_seq", 0)


def graph_payload(tickets: pd.DataFrame, clusters: pd.DataFrame, scored: pd.DataFrame | None) -> tuple[list, list]:
    """Nodes and links for the cluster map, built from the reviewer's working tables."""
    score_by = {}
    bucket_by = {}
    if scored is not None:
        score_by = dict(zip(scored["request_id"], scored["score"]))
        bucket_by = dict(zip(scored["request_id"], scored["bucket"].astype(str)))
    nodes, links = [], []

    def txt(v) -> str:
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    for _, c in clusters.iterrows():
        nodes.append({"id": str(c["cluster_id"]), "label": txt(c.get("label")) or str(c["cluster_id"]), "type": "cluster",
                      "effort": txt(c.get("effort_bucket")), "confidence": float(c.get("confidence") or 0),
                      "hypothesis": txt(c.get("root_cause_hypothesis")), "evidence": txt(c.get("evidence"))})
    cluster_ids = {n["id"] for n in nodes}
    for _, t in tickets.iterrows():
        nodes.append({"id": t["request_id"], "type": "ticket", "classification": t["classification"], "account": txt(t.get("source_account")),
                      "score": float(score_by.get(t["request_id"], 0) or 0), "bucket": bucket_by.get(t["request_id"], ""),
                      "summary": txt(t.get("summary")), "severity": txt(t.get("severity")), "confidence": float(t.get("confidence") or 0),
                      "effort": txt(t.get("effort_bucket")), "redirect": bool(t.get("redirect")), "metric_linked": bool(t.get("metric_linked")),
                      "reason": txt(t.get("reason")), "flag": txt(t.get("ambiguity_note")), "note": txt(t.get("reviewer_note")),
                      "cluster": txt(t.get("cluster_id"))})
        cid = str(t.get("cluster_id") or "").strip()
        if t["classification"] == "reactive" and cid and cid in cluster_ids:
            links.append({"source": t["request_id"], "target": cid, "kind": "member"})
        if t["classification"] == "proactive":
            ret = t.get("retires")
            ids = ret if isinstance(ret, list) else [x.strip() for x in str(ret or "").replace(";", ",").split(",") if x.strip()]
            for rid in ids:
                links.append({"source": t["request_id"], "target": rid, "kind": "retires"})
    return nodes, links


def apply_graph_edits(val: dict, tickets: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the graph's memberships, eliminates links and new clusters back into the working tables."""
    t = tickets.copy()
    c = clusters.copy()
    for nc in val.get("new_clusters", []):
        cid = str(nc.get("cluster_id", "")).strip()
        if cid and cid not in c["cluster_id"].astype(str).tolist():
            c = pd.concat([c, pd.DataFrame([{"cluster_id": cid, "label": nc.get("label", cid), "ticket_ids": "",
                                             "root_cause_hypothesis": "", "evidence": "added by reviewer on the map",
                                             "effort_bucket": "1-2 sprints", "confidence": 0.5}])], ignore_index=True)
    valid = set(c["cluster_id"].astype(str))
    for tid, cid in val.get("memberships", {}).items():
        mask = t["request_id"] == tid
        if mask.any() and (t.loc[mask, "classification"] == "reactive").all():
            t.loc[mask, "cluster_id"] = cid if cid in valid else ""
    for pid, ids in val.get("retires", {}).items():
        mask = t["request_id"] == pid
        if mask.any():
            t.loc[mask, "retires"] = ", ".join(ids)
    return t, c


def _clear(*keys: str) -> None:
    for k in keys:
        ss.pop(k, None)


def build_accounts_df(accounts: list[dict]) -> pd.DataFrame:
    present = sorted(set(ss.backlog["source_account"]) | {"Internal"})
    by_name = {a["source_account"]: a for a in accounts}
    rows = []
    for acct in present:
        a = by_name.get(acct, {})
        tier = a.get("tier", "internal" if acct == "Internal" else "unknown")
        sugg = int(a.get("suggested_weight", TIER_WEIGHTS.get(tier, 3)))
        rows.append(dict(
            account=acct, size_clue=tier, evidence=a.get("evidence", "no note found"),
            suggested_weight=sugg, weight=float(sugg), reason=a.get("weight_reason", ""),
            model_strategic=bool(a.get("strategic", False)), strategic=bool(a.get("strategic", False)),
            strategic_reason=a.get("strategic_reason", ""),
        ))
    return pd.DataFrame(rows)


def load_proposal(blob: dict) -> None:
    tk, cl = ai_pass.payload_to_frames(blob["result"])
    tk["retires"] = tk["retires"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))
    tk = tk.merge(ss.backlog[["request_id", "source_account", "summary", "date_received"]], on="request_id", how="left")
    tk["reviewer_note"] = ""
    ss.ai_blob = blob
    ss.ai_tickets = tk.copy()
    ss.ai_clusters = cl.copy()
    ss.tickets = tk.copy()
    ss.clusters = cl.copy()
    ss.accounts_df = build_accounts_df(blob["result"].get("accounts", []))
    ss.config["account_weight"] = dict(zip(ss.accounts_df["account"], ss.accounts_df["weight"]))
    ss.config["account_strategic"] = dict(zip(ss.accounts_df["account"], ss.accounts_df["strategic"]))
    ss.finalized = False
    _clear("ticket_editor", "cluster_editor", "account_editor")


def reset_backlog_state() -> None:
    ss.ai_blob = ss.ai_tickets = ss.ai_clusters = ss.tickets = ss.clusters = ss.accounts_df = None
    ss.finalized = False
    _clear("ticket_editor", "cluster_editor", "account_editor")


# --------------------------------------------------------------------------- #
# Sidebar: status, key, settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Status")
    st.write(f"File: {ss.backlog_name or 'none loaded'}")
    st.write("AI analysis: " + ("loaded" if ss.ai_blob is not None else "not yet"))
    st.write("Review: " + ("finalised" if ss.finalized else "open"))

    st.divider()
    st.header("AI model")
    server_key = ""
    try:
        server_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    server_key = server_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if server_key:
        api_key = server_key
        st.caption("API key is set on the server.")
    else:
        api_key = st.text_input("API key", type="password", help="Only needed to run the AI pass.")
    model = ai_pass.DEFAULT_MODEL
    st.caption(f"Model: {model}")

    st.divider()
    st.header("Settings")
    st.caption("The parameters behind the score. Change one and the results update. Account size is edited in step 3.")
    cfg = ss.config
    cfg["do_now_threshold"] = st.slider("Priority threshold (score required for 'Do now')", 0.5, 10.0, float(cfg["do_now_threshold"]), 0.25)
    cfg["quarter_capacity_points"] = st.number_input("Team capacity this quarter (effort points)", 1, 200, int(cfg["quarter_capacity_points"]))
    cfg["strategic_multiplier"] = st.slider("Strategic account multiplier (1.0 = off)", 1.0, 2.0, float(cfg["strategic_multiplier"]), 0.05,
                                            help="Applied to the impact of every ticket from an account marked strategic in step 3a.")
    cfg["metric_bonus"] = st.slider("Bonus for proactive items with a named metric (1.0 = off)", 1.0, 3.0, float(cfg["metric_bonus"]), 0.1)
    fill_labels = {"later": "Pull from Later", "later_and_declined": "Pull from Later and Not this quarter", "off": "Leave unallocated"}
    cfg["fill_spare_capacity"] = st.selectbox("Spare capacity", list(fill_labels), key="fill_mode",
                                              format_func=lambda k: fill_labels[k],
                                              help="After the threshold pass, the best-scoring items that fit are pulled into Do now until capacity is used.")
    with st.expander("More settings"):
        cfg["defer_ratio"] = st.slider("'Later' band (share of the threshold)", 0.1, 1.0, float(cfg["defer_ratio"]), 0.05)
        cfg["discovery_min_impact"] = st.slider("'Investigate first' (unclear effort) only if impact is at least", 0.0, 20.0, float(cfg["discovery_min_impact"]), 0.5)
        st.markdown("**Severity scale (multiplier, not work)**")
        for sv in SEVERITIES:
            cfg["severity_scale"][sv] = st.number_input(sv, 0.5, 10.0, float(cfg["severity_scale"][sv]), 0.5, key=f"sv_{sv}")
        st.markdown("**Effort points (same unit as team capacity)**")
        for eb in EFFORT_BUCKETS:
            label = {"unclear": "unclear (cost of finding out)"}.get(eb, eb)
            cfg["effort_points"][eb] = st.number_input(label, 0.5, 20.0, float(cfg["effort_points"][eb]), 0.5, key=f"ep_{eb}")
    if st.button("Reset settings"):
        for k in list(ss.keys()):
            if k.startswith(("ep_", "sv_")) or k == "fill_mode":
                del ss[k]
        aw = dict(ss.config["account_weight"]); strat = dict(ss.config.get("account_strategic", {}))
        ss.config = copy.deepcopy(DEFAULT_CONFIG)
        if ss.accounts_df is not None:
            ss.accounts_df["weight"] = ss.accounts_df["suggested_weight"].astype(float)
            ss.accounts_df["strategic"] = ss.accounts_df["model_strategic"].astype(bool)
            aw = dict(zip(ss.accounts_df["account"], ss.accounts_df["weight"]))
            strat = dict(zip(ss.accounts_df["account"], ss.accounts_df["strategic"]))
            _clear("account_editor")
        ss.config["account_weight"] = aw
        ss.config["account_strategic"] = strat
        st.rerun()

config = merged_config(ss.config)

# --------------------------------------------------------------------------- #
# Title and how it works
# --------------------------------------------------------------------------- #
st.title("Backlog scorer")
st.markdown("Reads a backlog of inbound requests and internal proposals, scores each item, and recommends how much of next quarter "
            "should go to reactive work (customer requests) versus proactive work (team initiatives).")

with st.container(border=True):
    st.markdown("""
**How this works**

1. **Load the backlog.** Upload the CSV or use the bundled sample.
2. **AI analysis.** The model reads every ticket together and proposes: which tickets share one root cause, the severity of each, our confidence in the cause, the effort involved, each account's size, and which proactive items would eliminate existing requests. It does not score anything.
3. **Review and adjust.** You make the final call. Change any proposal you disagree with and record why. Add clusters the model missed. Adjust the settings on the left.
4. **Results and export.** The scoring runs on your final tags. You get the recommended split, the rationale, the assumptions, and a log of every change you made.

**Please review before relying on the output.** The model's proposals are an interpretation of free text and can be wrong: it can cluster tickets that are different bugs, set severity from the tone of an email instead of the figures, misjudge an account's size, or credit a proactive item with resolving something it would only detect. The recommendation is arithmetic on those tags. It is only as sound as the review in step 3.
""")

with st.expander("How the score works"):
    c = config
    d = DEFAULT_CONFIG

    def cur(key, fmt="{:g}"):
        """Current value, with the default in brackets when it has been changed."""
        v, dv = c[key], d[key]
        return fmt.format(v) if v == dv else f"{fmt.format(v)} (default {fmt.format(dv)})"

    st.markdown(f"""
Every ticket receives a single score. Higher means higher priority.

*The numbers below are the current settings. Each can be changed in the sidebar; where a value has been changed, the default is shown in brackets.*

**score = impact × confidence ÷ effort**

- **Impact** = account size (1 to 5, suggested by the model, editable in step 3) × severity{" × " + format(c["strategic_multiplier"], "g") + " for strategic accounts" if c["strategic_multiplier"] != 1 else ""}. Account size is the revenue tier only. A **strategic** account is one that matters beyond its tier (fast growth, reference customer, regulatory or partnership relationship, market entry): an account-level, forward-looking fact, proposed by the model and editable in step 3. Churn risk, escalations and executive involvement are per-ticket facts and are captured by severity, so they are never folded into size or the strategic flag. Severity is a multiplier, not a unit of work: low {c["severity_scale"]["low"]:g}, medium {c["severity_scale"]["medium"]:g}, high {c["severity_scale"]["high"]:g}. Note the ranges: account size runs 1 to 5 and severity 1 to 3, so a large account's minor issue can outrank a small account's serious one. That is a deliberate bias towards revenue at risk; widen the severity scale in the sidebar if you want severity to carry more.
- **Confidence** is 0 to 1: how well do we know the cause and the fix?
- **Effort** is in points of work: small {c["effort_points"]["small"]:g}, 1-2 sprints {c["effort_points"]["1-2 sprints"]:g}, large {c["effort_points"]["large"]:g}. The team's rough_effort_hint is taken as the estimate and trusted by default; the model overrides it only when something specific in the ticket contradicts it, and every override is flagged for review. "Unclear" is never scored as if it were known: it goes to Investigate first (costing {c["effort_points"]["unclear"]:g} point) or, if the impact is too small to justify that, to Not this quarter. "Large" counts as this quarter's slice of a multi-quarter build. Team capacity is measured in these same effort points, so allocation is simply adding effort until capacity is reached.

**Tickets that share one root cause are scored as a cluster.** Their impact is summed, the effort is counted once, and every ticket in the cluster receives the cluster's score. This is why clustering matters: one fix that closes three tickets is worth three tickets. A cluster member that has been handed off adds no impact and carries no cost, since the engineering fix is not what resolves it.

**Proactive items** receive their own impact, plus the impact of every reactive ticket they would eliminate{" (times a " + format(c["metric_bonus"], "g") + "× bonus for items with a named metric)" if c["metric_bonus"] != 1 else ""}. That is how a structural fix can outrank a single loud request. It also makes the "eliminates" list the single biggest lever on the proactive share: it is a hypothesis about what the item would prevent, and the reviewer should treat it as one. The metric bonus is off by default because urgency is already captured in severity; turning it on rewards measurability.

**Each ticket is then assigned to one of five categories, checked in this order:**

1. **Hand off.** Not engineering work (configuration, data cleanup, process). Reassigned to the owning team. Excluded from capacity.
2. **Investigate first.** Effort is unclear and impact is at least {c["discovery_min_impact"]:g}. A short, time-boxed investigation before estimating. Unclear effort with lower impact goes to Not this quarter.
3. **Do now.** Score of {cur("do_now_threshold")} or more (the priority threshold, set in the sidebar).
4. **Later.** Score of {round(c["do_now_threshold"] * c["defer_ratio"], 2):g} or more; deferred, revisited if capacity allows.
5. **Not this quarter.** Everything else; declined for this cycle.

Capacity is then allocated: "Do now" items are taken in score order until {cur("quarter_capacity_points")} points are used (the team capacity, set under "Team capacity this quarter" in the sidebar). Anything that does not fit moves to "Later". If points remain, the best-scoring items from "Later" (and, if enabled, "Not this quarter") that fit are pulled up into "Do now", so capacity is not left unused. Those items carry a "pulled up" note in the results and the rationale says they were funded on spare capacity, not on merit. This means the priority threshold is a quality floor, not the thing that decides funding: capacity decides funding, the threshold decides what is good enough to be pulled up.

**The split** is the share of allocated effort points going to reactive versus proactive work. Handed-off work is excluded because it does not consume engineering capacity.

| Setting | Where | Now | What changing it does |
|---|---|---|---|
| Strategic multiplier | Sidebar | {c["strategic_multiplier"]:g} | Applied to every ticket from an account marked strategic in step 3a. Off at 1.0. |
| Account size | Step 3 | {", ".join(f"{k} {v:g}" for k, v in sorted(c["account_weight"].items()))} | The single biggest lever. Bigger accounts push their tickets up. |
| Priority threshold | Sidebar | {cur("do_now_threshold")} | The quality floor. Lower it and more qualifies, usually more proactive work. |
| Team capacity | Sidebar | {cur("quarter_capacity_points")} | Effort points available this quarter, same unit as effort. The default assumes five people, six two-week sprints, about five points per sprint after support load. |
| Metric bonus | Sidebar | {c["metric_bonus"]:g} | Off at 1.0. Raise it to favour proactive items with a named metric. |
| Spare capacity | Sidebar | {c["fill_spare_capacity"]} | Whether leftover capacity is filled from lower categories or left unallocated. |
| 'Later' band | Sidebar | {c["defer_ratio"]:g} | How far below the threshold still qualifies as "Later" rather than "Not this quarter". |
| Investigation floor | Sidebar | impact at least {c["discovery_min_impact"]:g} | Unclear-effort items above this get an investigation; below it they are declined for the quarter. |
| Severity scale, effort points | Sidebar | see above | Widen the gaps to make severity or effort count more. Severity is a multiplier; effort and capacity share a unit. |
| Per-ticket tags | Step 3 | | Cluster, severity, confidence, effort, hand off, metric-linked, eliminates. The model proposes, you decide. |
| Clusters | Step 3 | | Effort and confidence for the shared fix. Override the tickets' own values. |
""")

# --------------------------------------------------------------------------- #
# Step 1: load
# --------------------------------------------------------------------------- #
st.header("Step 1. Load the backlog")
c1, c2 = st.columns([2, 1])
with c1:
    uploaded = st.file_uploader("CSV with: request_id, date_received, source_account, request_type, summary, raw_notes, rough_effort_hint", type=["csv"])
with c2:
    st.write(""); st.write("")
    use_bundled = st.button("Use the bundled Trellis backlog", disabled=not DATA_PATH.exists())

if uploaded is not None and ss.backlog_name != uploaded.name:
    reset_backlog_state()
    ss.backlog = read_backlog(uploaded)
    ss.backlog_name = uploaded.name
    st.rerun()
elif use_bundled:
    if ss.backlog_name != DATA_PATH.name:
        reset_backlog_state()
    ss.backlog = read_backlog(DATA_PATH)
    ss.backlog_name = DATA_PATH.name
    st.rerun()


def empty_sections(from_step: int, why: str) -> None:
    titles = {2: "Step 2. AI analysis", 3: "Step 3. Review and adjust", 4: "Step 4. Results", 5: "Step 5. Finalise and export"}
    for n in range(from_step, 6):
        st.header(titles[n])
        st.caption(why)


if ss.backlog is None:
    st.info("Nothing loaded yet. Upload a CSV or use the bundled backlog.")
    with st.expander("The backlog"):
        st.caption("Empty until a file is loaded.")
    empty_sections(2, "Waiting for a backlog.")
    st.stop()

bl = ss.backlog
st.caption(f"{ss.backlog_name}: {len(bl)} tickets from {bl['source_account'].nunique()} sources. "
           f"Effort hints read as: { {k: int(v) for k, v in bl['effort_bucket'].value_counts().items()} }.")
with st.expander("The backlog"):
    st.dataframe(bl.drop(columns=["effort_bucket"]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Step 2: AI pass
# --------------------------------------------------------------------------- #
st.header("Step 2. AI analysis")
st.markdown("A single call with the entire backlog. The model proposes tags and explains each one. It is instructed not to score.")

with st.expander("Model instructions (editable before running)"):
    st.caption("Adjust the wording for different definitions or emphasis. The fields the model must return are fixed.")
    edited_prompt = st.text_area("prompt", value=ss.system_prompt, height=400, key="prompt_editor", label_visibility="collapsed")
    p1, p2, p3 = st.columns([1, 1, 3])
    if p1.button("Use this version", disabled=edited_prompt == ss.system_prompt):
        ss.system_prompt = edited_prompt
        st.rerun()
    if p2.button("Back to default", disabled=ss.system_prompt == ai_pass.SYSTEM_PROMPT):
        ss.system_prompt = ai_pass.SYSTEM_PROMPT
        _clear("prompt_editor")
        st.rerun()
    if ss.system_prompt != ai_pass.SYSTEM_PROMPT:
        p3.caption("Custom instructions in use. Runs with them are saved separately.")
    with st.popover("Fields the model must return"):
        st.json(ai_pass.TOOL_SCHEMA["input_schema"], expanded=False)

cached = ai_pass.load_cache(bl, model, ss.system_prompt)
cur_hash = ai_pass.prompt_hash(bl, model, ss.system_prompt)
runs_left = MAX_RUNS_PER_SESSION - ss.ai_runs

b1, b2, b3 = st.columns([1, 1, 2])
run_api = b1.button(f"Run AI analysis ({runs_left} left this session)", type="primary", disabled=(not api_key) or runs_left <= 0)
load_cached = b2.button("Load saved analysis", disabled=cached is None)
b3.caption(f"Model: {model}" + ("  |  custom instructions" if ss.system_prompt != ai_pass.SYSTEM_PROMPT else ""))

if run_api:
    with st.spinner("Reading the backlog..."):
        try:
            blob = ai_pass.run_ai_pass(bl, api_key, model, ss.system_prompt)
            ss.ai_runs += 1
            ai_pass.save_cache(blob)
            load_proposal(blob)
            st.toast(f"Done. Tokens used: {blob['usage']}")
            st.rerun()
        except Exception as e:
            st.error(f"The AI pass failed: {e}")

if load_cached and cached:
    load_proposal(cached)
    st.rerun()

if ss.ai_blob is None:
    if cached is None:
        st.info("No saved analysis for this file yet. Run the AI analysis to generate one.")
    else:
        st.caption(f"A saved analysis exists ({cached.get('source')}, {cached.get('generated_at')}). "
                   + ("It matches the current file and instructions." if cached.get("prompt_hash") == cur_hash
                      else "The file or instructions have changed since; consider running again."))
    empty_sections(3, "Waiting for the AI analysis.")
    st.stop()

meta = ss.ai_blob
if meta.get("source") == "seed":
    st.warning("This is the sample analysis that ships with the app, written offline, not a live model run. Run the AI analysis to replace it.")
else:
    st.caption(f"Analysis from {meta.get('model')} at {meta.get('generated_at')}.")

acc_raw = meta["result"].get("accounts", [])
if any("no entry that matched" in str(a.get("evidence", "")) for a in acc_raw):
    st.warning("The model's account list did not line up with the account names in the file for at least one account. "
               "Those accounts were given a default size of 3. Check step 3a, edit the sizes, or run the analysis again.")

obs = meta["result"].get("cross_record_observations", [])
if obs:
    with st.expander("Cross-ticket observations from the model", expanded=True):
        for o in obs:
            st.markdown(f"- {o}")

# --------------------------------------------------------------------------- #
# Step 3: review
# --------------------------------------------------------------------------- #
st.header("Step 3. Review and adjust")
locked = ss.finalized
if locked:
    st.success("Finalised. Tables are locked. Use 'Reopen' in step 5 to make changes.")
else:
    st.markdown("Everything below is a proposal until you finalise. Change what you disagree with and add a note so the change is recorded in the log.")

# ---- 3a accounts ----
st.subheader("3a. Account size")
st.caption("The model read the notes for size indicators and proposed an account size from 1 (small) to 5 (top). Size is the revenue tier only. "
           "Separately, an account can be marked strategic when it matters beyond its tier (fast growth, reference customer, regulatory or partnership relationship, market entry); "
           f"its tickets' impact is multiplied by {config['strategic_multiplier']:g} (sidebar). Churn risk and escalations are per-ticket and handled by severity. Edit either column if you disagree.")
acc_editor = st.data_editor(
    ss.accounts_df,
    key="account_editor",
    disabled=locked or ["account", "size_clue", "evidence", "suggested_weight", "reason", "model_strategic", "strategic_reason"],
    hide_index=True,
    width="stretch",
    column_order=["account", "size_clue", "suggested_weight", "weight", "model_strategic", "strategic", "strategic_reason", "reason", "evidence"],
    column_config={
        "model_strategic": st.column_config.CheckboxColumn("model: strategic", width="small"),
        "strategic": st.column_config.CheckboxColumn("strategic (edit)", width="small"),
        "strategic_reason": st.column_config.TextColumn("strategic reason", width="medium"),
        "account": st.column_config.TextColumn("account", width="medium"),
        "size_clue": st.column_config.TextColumn("size per notes", width="small"),
        "suggested_weight": st.column_config.NumberColumn("model proposes", width="small"),
        "weight": st.column_config.NumberColumn("account size (edit)", min_value=0.0, max_value=10.0, step=0.5, format="%.1f", width="small"),
        "reason": st.column_config.TextColumn("model's reason", width="large"),
        "evidence": st.column_config.TextColumn("evidence in notes", width="large"),
    },
)
working_accounts = ss.accounts_df if locked else acc_editor
ss.config["account_weight"] = {a: float(w) for a, w in zip(working_accounts["account"], working_accounts["weight"].fillna(3.0))}
ss.config["account_strategic"] = {a: bool(v) for a, v in zip(working_accounts["account"], working_accounts["strategic"].fillna(False))}
config = merged_config(ss.config)

# ---- 3b clusters ----
st.subheader("3b. Clusters: tickets that share one root cause")
st.caption("A cluster's effort and confidence apply to every ticket in it. Membership is set per ticket in 3c; the tickets column here is the model's original proposal.")

with st.expander("Add a cluster the model missed"):
    a1, a2 = st.columns([1, 2])
    new_id = a1.text_input("cluster id (short, no spaces)", key="new_cl_id", disabled=locked)
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
                "root_cause_hypothesis": new_hyp.strip(), "evidence": "added by reviewer",
                "effort_bucket": new_effort, "confidence": float(new_conf),
            }])
            ss.clusters = pd.concat([ss.get("_pending_clusters", ss.clusters), row], ignore_index=True)
            _clear("cluster_editor")
            st.rerun()

cl_editor = st.data_editor(
    ss.clusters,
    key="cluster_editor",
    disabled=locked or ["cluster_id", "ticket_ids", "evidence", "root_cause_hypothesis"],
    hide_index=True,
    width="stretch",
    column_order=["cluster_id", "label", "ticket_ids", "effort_bucket", "confidence", "root_cause_hypothesis", "evidence"],
    column_config={
        "cluster_id": st.column_config.TextColumn("cluster", width="small"),
        "label": st.column_config.TextColumn("label", width="medium"),
        "ticket_ids": st.column_config.TextColumn("tickets (model's proposal)", width="small"),
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
        _clear("cluster_editor", "ticket_editor", "rm_cl")
        st.rerun()

# ---- 3c tickets ----
st.subheader("3c. Tickets")
st.caption("Select a cluster to assign a ticket to it. For proactive items, list the request ids it would eliminate, separated by commas.")
tk_editor = st.data_editor(
    ss.tickets,
    key="ticket_editor",
    disabled=locked or ["request_id", "source_account", "summary", "date_received", "reason", "ambiguity_note"],
    hide_index=True,
    width="stretch",
    height=620,
    column_order=[
        "request_id", "source_account", "classification", "cluster_id", "severity", "confidence",
        "effort_bucket", "redirect", "metric_linked", "retires", "reviewer_note", "reason", "ambiguity_note", "summary",
    ],
    column_config={
        "request_id": st.column_config.TextColumn("ticket", width="small"),
        "source_account": st.column_config.TextColumn("account", width="small"),
        "classification": st.column_config.SelectboxColumn("reactive / proactive", options=CLASSIFICATIONS, width="small"),
        "cluster_id": st.column_config.SelectboxColumn("cluster", options=cluster_ids, width="small"),
        "severity": st.column_config.SelectboxColumn("severity", options=SEVERITIES, width="small"),
        "confidence": st.column_config.NumberColumn("confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
        "effort_bucket": st.column_config.SelectboxColumn("effort", options=EFFORT_BUCKETS, width="small"),
        "redirect": st.column_config.CheckboxColumn("hand off", width="small"),
        "metric_linked": st.column_config.CheckboxColumn("metric-linked", width="small"),
        "retires": st.column_config.TextColumn("eliminates", width="medium"),
        "reviewer_note": st.column_config.TextColumn("reviewer note", width="medium"),
        "reason": st.column_config.TextColumn("model's reason", width="large"),
        "ambiguity_note": st.column_config.TextColumn("flagged for review", width="large"),
        "summary": st.column_config.TextColumn("summary", width="large"),
    },
)
working_tickets = ss.tickets if locked else tk_editor
ss._pending_tickets = working_tickets

# ---- 3d cluster map ----
st.subheader("3d. Cluster map")
st.caption("The same clusters and tickets as the tables above, as a map. Click a node for its details. Scroll to zoom, drag the background to pan, drag nodes to rearrange. "
           + ("Locked while finalised." if locked else "Drag a reactive ticket onto a cluster to assign it, drag a proactive ticket onto a reactive one to mark it as eliminated, click a line to remove it, double-click empty space to add a cluster. Edits update the tables."))
try:
    _preview_scored, _ = score_backlog(working_tickets, working_clusters, config)
except Exception:
    _preview_scored = None
g_nodes, g_links = graph_payload(working_tickets, working_clusters, _preview_scored)
graph_val = _cluster_graph(nodes=g_nodes, links=g_links, locked=locked, height=640, key="cluster_graph", default=None)
if graph_val and not locked and int(graph_val.get("seq", 0)) > int(ss.graph_seq):
    ss.graph_seq = int(graph_val["seq"])
    new_t, new_c = apply_graph_edits(graph_val, working_tickets, working_clusters)
    ss.tickets, ss.clusters = new_t, new_c
    _clear("ticket_editor", "cluster_editor")
    st.rerun()

if len(working_clusters):
    counts = working_tickets["cluster_id"].fillna("").astype(str).value_counts()
    lonely = [c for c in working_clusters["cluster_id"].astype(str) if counts.get(c, 0) < 2]
    if lonely:
        st.caption(f"Clusters with fewer than two tickets (scored as single tickets until more are assigned): {', '.join(lonely)}")
unknown = sorted(set(working_tickets["cluster_id"].fillna("").astype(str)) - set(cluster_ids))
if unknown:
    st.warning(f"Some tickets reference clusters that no longer exist: {', '.join(unknown)}. They will be scored on their own values.")

if not locked and st.button("Discard my edits and restore the model's proposals"):
    ss.tickets = ss.ai_tickets.copy()
    ss.clusters = ss.ai_clusters.copy()
    ss.accounts_df = build_accounts_df(meta["result"].get("accounts", []))
    _clear("ticket_editor", "cluster_editor", "account_editor")
    st.rerun()

# --------------------------------------------------------------------------- #
# Step 4: results
# --------------------------------------------------------------------------- #
st.header("Step 4. Results" + ("" if locked else "  (live preview; updates as you edit)"))

try:
    scored, clusters_scored = score_backlog(working_tickets, working_clusters, config)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

split = capacity_split(scored, config)
diff = diff_tags(ss.ai_tickets, working_tickets)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Reactive", f"{split['reactive_pct']}%")
m1.caption(f"{split['reactive_points']:g} effort pts · customer requests")
m2.metric("Proactive", f"{split['proactive_pct']}%")
m2.caption(f"{split['proactive_points']:g} effort pts · team initiatives")
m3.metric("Allocated", f"{split['committed_points']:g}")
m3.caption(f"of {split['capacity_points']:g} effort pts capacity")
m4.metric("Unallocated", f"{split['headroom_points']:g}")
m4.caption("effort pts")
m5.metric("Reactive by count", f"{split['reactive_in_backlog_pct']}%")
m5.caption("share of tickets, for comparison")
m6.metric("Changed", f"{len(diff)}")
m6.caption(f"proposals, on {diff['request_id'].nunique()} tickets" if len(diff) else "proposals")

st.subheader("Rationale for the split")
st.markdown(explain_split(scored, clusters_scored, split, config, working_tickets, working_clusters))

DISPLAY_COLS = {
    "request_id": "ticket", "source_account": "account", "classification": "type", "cluster_id": "cluster",
    "severity": "severity", "eff_confidence": "confidence", "eff_effort_bucket": "effort", "impact": "impact",
    "cost_share": "effort pts", "score": "score", "capacity_note": "note", "reason": "rationale",
}


def show(sub: pd.DataFrame) -> None:
    st.dataframe(sub[list(DISPLAY_COLS)].rename(columns=DISPLAY_COLS), width="stretch", hide_index=True,
                 column_config={"rationale": st.column_config.TextColumn("rationale", width="large")})


tab_labels = BUCKET_ORDER + ["Clusters", "Assumptions", "Judgement calls", "Changes from the model", "All tickets"]
tabs = st.tabs(tab_labels)
for tab, bucket in zip(tabs[: len(BUCKET_ORDER)], BUCKET_ORDER):
    with tab:
        st.caption(BUCKET_HELP[bucket])
        sub = scored[scored["bucket"] == bucket]
        if sub.empty:
            st.write("Nothing here.")
        else:
            show(sub)
with tabs[len(BUCKET_ORDER)]:
    st.caption("One row per fix. Unclustered tickets appear as their own row.")
    st.dataframe(clusters_scored.rename(columns={"key": "cluster", "label": "label", "breadth": "tickets in cluster",
                                                 "effort_bucket": "effort", "confidence": "confidence", "cluster_impact": "impact",
                                                 "cost_points": "points"}), width="stretch", hide_index=True)
with tabs[len(BUCKET_ORDER) + 1]:
    st.markdown(list_assumptions(scored, clusters_scored, config, meta["result"].get("accounts", []), meta, diff))
with tabs[len(BUCKET_ORDER) + 2]:
    gaps = gaps_table(scored, working_tickets)
    st.caption("Every ticket the model flagged as needing a judgement, in one place: what was flagged, the tags it ended up with, where it landed, and your note. "
               "Add a reviewer note in step 3c to record your reasoning here.")
    if gaps.empty:
        st.write("No tickets were flagged.")
    else:
        st.write(f"{len(gaps)} of {len(scored)} tickets flagged.")
        st.dataframe(gaps, width="stretch", hide_index=True,
                     column_config={"what was flagged": st.column_config.TextColumn(width="large"),
                                    "call made": st.column_config.TextColumn(width="large"),
                                    "reviewer note": st.column_config.TextColumn(width="medium")})
with tabs[len(BUCKET_ORDER) + 3]:
    if diff.empty:
        st.write("No changes from the model's proposals.")
    else:
        st.dataframe(diff.rename(columns={"ai_proposed": "model proposed", "reviewer_final": "reviewer final", "reviewer_note": "reviewer note"}),
                     width="stretch", hide_index=True)
with tabs[len(BUCKET_ORDER) + 4]:
    st.dataframe(scored.drop(columns=["summary"], errors="ignore"), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Step 5: finalise
# --------------------------------------------------------------------------- #
st.header("Step 5. Finalise and export")
f1, f2, f3 = st.columns([1, 1, 4])
if f1.button("Finalise", type="primary", disabled=locked):
    ss.tickets = working_tickets.copy()
    ss.clusters = working_clusters.copy()
    ss.accounts_df = working_accounts.copy()
    _clear("ticket_editor", "cluster_editor", "account_editor")
    ss.finalized = True
    st.rerun()
if f2.button("Reopen", disabled=not locked):
    ss.finalized = False
    st.rerun()

if not locked:
    st.caption("Finalise locks the tables and enables the downloads.")
else:
    summary_md = build_summary(scored, clusters_scored, split, config, diff, meta, obs, working_tickets, working_clusters)
    final_json = {
        "config": config,
        "ai_pass": {k: meta.get(k) for k in ("source", "model", "generated_at", "prompt_hash")},
        "accounts": json.loads(working_accounts.to_json(orient="records")),
        "tickets": json.loads(working_tickets.to_json(orient="records")),
        "clusters": json.loads(working_clusters.to_json(orient="records")),
    }
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("Scored tickets (CSV)", scored.to_csv(index=False), "scored_backlog.csv", "text/csv")
    e2.download_button("Summary (markdown)", summary_md, "summary.md", "text/markdown")
    e3.download_button("Changes from the model (CSV)", diff.to_csv(index=False), "changes_from_model.csv", "text/csv")
    e4.download_button("Final tags and settings (JSON)", json.dumps(final_json, indent=2), "finalized.json", "application/json")
    with st.expander("Preview the summary"):
        st.markdown(summary_md)
