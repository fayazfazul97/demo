"""
Backlog scorer: customer requests vs our own initiatives.

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

import ai_pass
from report import build_summary, explain_split, list_assumptions
from scoring import (
    BUCKET_HELP,
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
MAX_RUNS_PER_SESSION = 3

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
    st.write("AI proposal: " + ("loaded" if ss.ai_blob is not None else "not yet"))
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
    st.caption("Plain numbers behind the score. Change one and the results update. Account weights are edited in step 3.")
    cfg = ss.config
    cfg["do_now_threshold"] = st.slider("Funding line (score needed for 'Do now')", 0.5, 10.0, float(cfg["do_now_threshold"]), 0.25)
    cfg["quarter_capacity_points"] = st.number_input("Team capacity this quarter (points)", 1, 200, int(cfg["quarter_capacity_points"]))
    cfg["metric_bonus"] = st.slider("Bonus for initiatives tied to a company goal", 1.0, 3.0, float(cfg["metric_bonus"]), 0.1)
    with st.expander("More settings"):
        cfg["defer_ratio"] = st.slider("'Later' if score is at least this share of the line", 0.1, 1.0, float(cfg["defer_ratio"]), 0.05)
        cfg["discovery_confidence_max"] = st.slider("'Investigate first' when how-sure is below", 0.0, 1.0, float(cfg["discovery_confidence_max"]), 0.05)
        cfg["discovery_min_impact"] = st.slider("...and 'matters' is at least", 0.0, 20.0, float(cfg["discovery_min_impact"]), 0.5)
        st.markdown("**Points for how serious**")
        for sv in SEVERITIES:
            cfg["severity_scale"][sv] = st.number_input(sv, 0.5, 10.0, float(cfg["severity_scale"][sv]), 0.5, key=f"sv_{sv}")
        st.markdown("**Points for how much work**")
        for eb in EFFORT_BUCKETS:
            label = {"unclear": "unclear (cost of finding out)"}.get(eb, eb)
            cfg["effort_points"][eb] = st.number_input(label, 0.5, 20.0, float(cfg["effort_points"][eb]), 0.5, key=f"ep_{eb}")
    if st.button("Reset settings"):
        for k in list(ss.keys()):
            if k.startswith(("ep_", "sv_")):
                del ss[k]
        aw = dict(ss.config["account_weight"])
        ss.config = copy.deepcopy(DEFAULT_CONFIG)
        if ss.accounts_df is not None:
            ss.accounts_df["weight"] = ss.accounts_df["suggested_weight"].astype(float)
            aw = dict(zip(ss.accounts_df["account"], ss.accounts_df["weight"]))
            _clear("account_editor")
        ss.config["account_weight"] = aw
        st.rerun()

config = merged_config(ss.config)

# --------------------------------------------------------------------------- #
# Title and how it works
# --------------------------------------------------------------------------- #
st.title("Backlog scorer")
st.markdown("Reads a backlog of requests and ideas, scores each one, and recommends how much of next quarter should go to "
            "customer requests versus the team's own initiatives.")

with st.container(border=True):
    st.markdown("""
**How it works, in four steps**

1. **Load the backlog.** Upload the CSV or use the bundled one.
2. **Let the AI model read it.** It reads every ticket together and suggests: which tickets share one fix, how serious each is, how sure we are about the cause, how much work it is, how important each account is, and which initiatives would make requests go away. It does not score anything.
3. **Review and adjust.** You are the judge. Change any suggestion you disagree with and say why. Add groups the model missed. Adjust the settings on the left.
4. **See the results and download.** The maths runs on your final tags. You get the split, the reasoning, the assumptions, and a log of everything you changed.

**Please review before you trust the output.** The model's suggestions are a reading of free text and can be wrong: it can group tickets that are different bugs, rate a ticket on the tone of the email instead of the numbers, misjudge an account's size, or credit an initiative with fixing something it would only detect. The recommendation is arithmetic on those tags. It is only as good as the review in step 3.
""")

with st.expander("How the score works (plain version)"):
    c = config
    st.markdown(f"""
Every ticket gets one number, its score. Higher means fund it sooner.

**score = how much it matters × how sure we are ÷ how much work it is**

- **How much it matters** = the account's importance (1 to 5, suggested by the model, editable in step 3) × how serious the problem is (low {c["severity_scale"]["low"]:g}, medium {c["severity_scale"]["medium"]:g}, high {c["severity_scale"]["high"]:g}).
- **How sure we are** is 0 to 1: do we know the cause and the fix?
- **How much work** is in points: small {c["effort_points"]["small"]:g}, 1-2 sprints {c["effort_points"]["1-2 sprints"]:g}, large {c["effort_points"]["large"]:g}. "Unclear" is not guessed; it costs {c["effort_points"]["unclear"]:g} point to find out.

**Tickets that share one fix are scored as a group.** Their "matters" points are added up, the work is counted once, and every ticket in the group gets the group's score. This is why grouping matters: one fix that closes three tickets is worth three tickets.

**Our own initiatives** get their own "matters" points, plus the points of every request they would make go away for good, times a {c["metric_bonus"]:g}× bonus if they are tied to a measurable company goal. That is how a structural fix can beat a single loud request.

**Then every ticket lands in one of five piles, checked in this order:**

1. **Hand off.** Not engineering work (config, data cleanup, process). Goes to whoever owns that. Not counted.
2. **Investigate first.** Cause unknown, how-sure below {c["discovery_confidence_max"]:g}, and it matters at least {c["discovery_min_impact"]:g}. Spend a little time finding out.
3. **Do now.** Score of {c["do_now_threshold"]:g} or more (the funding line).
4. **Later.** Score of {round(c["do_now_threshold"] * c["defer_ratio"], 2):g} or more.
5. **Not this quarter.** Everything else.

Then we fill the quarter: "Do now" items are taken best-first until {c["quarter_capacity_points"]:g} points are used. Anything that does not fit moves to "Later".

**The split** is the share of those points going to customer requests versus our own initiatives. Handed-off work is left out because it does not use engineering time.

| Setting | Where | Now | What changing it does |
|---|---|---|---|
| Account weight | Step 3 | {", ".join(f"{k} {v:g}" for k, v in sorted(c["account_weight"].items()))} | The single biggest lever. Bigger accounts push their tickets up. |
| Funding line | Sidebar | {c["do_now_threshold"]:g} | Lower it and more gets funded, usually more initiatives. |
| Team capacity | Sidebar | {c["quarter_capacity_points"]:g} | How many points fit. |
| Bonus for a company goal | Sidebar | {c["metric_bonus"]:g} | Set to 1 to remove the initiative advantage. |
| 'Later' band | Sidebar | {c["defer_ratio"]:g} | How far below the line still counts as "Later" rather than "Not this quarter". |
| Investigate rules | Sidebar | below {c["discovery_confidence_max"]:g}, at least {c["discovery_min_impact"]:g} | When an unknown is worth looking into. |
| How serious, how much work | Sidebar | see above | Widen the gaps to make severity or effort count more. |
| Per-ticket tags | Step 3 | | Group, serious, sure, work, hand off, goal, makes-go-away. The model suggests, you decide. |
| Groups | Step 3 | | Work and how-sure for the shared fix. Override the tickets' own values. |
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
    titles = {2: "Step 2. Let the AI model read it", 3: "Step 3. Review and adjust", 4: "Step 4. Results", 5: "Step 5. Finalise and download"}
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
           f"Work-size hints read as: {dict(bl['effort_bucket'].value_counts())}.")
with st.expander("The backlog"):
    st.dataframe(bl.drop(columns=["effort_bucket"]), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Step 2: AI pass
# --------------------------------------------------------------------------- #
st.header("Step 2. Let the AI model read it")
st.markdown("One call with the whole backlog. The model suggests tags and explains each one. It is told not to score.")

with st.expander("The instructions the model gets (you can edit them before running)"):
    st.caption("Change the wording if you want different definitions or emphasis. The fields it must return are fixed.")
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
run_api = b1.button(f"Run the AI model ({runs_left} left this session)", type="primary", disabled=(not api_key) or runs_left <= 0)
load_cached = b2.button("Load the saved result", disabled=cached is None)
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
        st.info("No saved result for this file yet. Run the model to get one.")
    else:
        st.caption(f"A saved result exists ({cached.get('source')}, {cached.get('generated_at')}). "
                   + ("It matches the current file and instructions." if cached.get("prompt_hash") == cur_hash
                      else "The file or instructions have changed since; consider running again."))
    empty_sections(3, "Waiting for the AI model's suggestions.")
    st.stop()

meta = ss.ai_blob
if meta.get("source") == "seed":
    st.warning("This is the sample result that ships with the app, written offline, not a live model run. Run the model to replace it.")
else:
    st.caption(f"Result from {meta.get('model')} at {meta.get('generated_at')}.")

obs = meta["result"].get("cross_record_observations", [])
if obs:
    with st.expander("What the model noticed across tickets", expanded=True):
        for o in obs:
            st.markdown(f"- {o}")

# --------------------------------------------------------------------------- #
# Step 3: review
# --------------------------------------------------------------------------- #
st.header("Step 3. Review and adjust")
locked = ss.finalized
if locked:
    st.success("Finalised. Tables are locked. Use 'Reopen' in step 5 to change anything.")
else:
    st.markdown("Everything below is a suggestion until you finalise. Change what you disagree with and add a note so the change is explained in the log.")

# ---- 3a accounts ----
st.subheader("3a. How much each account counts")
st.caption("The model read the notes for size clues and suggested a weight from 1 (small) to 5 (top). Edit the weight column if you disagree.")
acc_editor = st.data_editor(
    ss.accounts_df,
    key="account_editor",
    disabled=locked or ["account", "size_clue", "evidence", "suggested_weight", "reason"],
    hide_index=True,
    width="stretch",
    column_order=["account", "size_clue", "suggested_weight", "weight", "reason", "evidence"],
    column_config={
        "account": st.column_config.TextColumn("account", width="medium"),
        "size_clue": st.column_config.TextColumn("size read from notes", width="small"),
        "suggested_weight": st.column_config.NumberColumn("model suggests", width="small"),
        "weight": st.column_config.NumberColumn("weight (edit)", min_value=0.0, max_value=10.0, step=0.5, format="%.1f", width="small"),
        "reason": st.column_config.TextColumn("model's reason", width="large"),
        "evidence": st.column_config.TextColumn("evidence in notes", width="large"),
    },
)
working_accounts = ss.accounts_df if locked else acc_editor
ss.config["account_weight"] = {a: float(w) for a, w in zip(working_accounts["account"], working_accounts["weight"].fillna(3.0))}
config = merged_config(ss.config)

# ---- 3b groups ----
st.subheader("3b. Groups: tickets that share one fix")
st.caption("A group's work and how-sure apply to every ticket in it. Which tickets belong is set per ticket in 3c; the tickets column here is the model's original suggestion.")

with st.expander("Add a group the model missed"):
    a1, a2 = st.columns([1, 2])
    new_id = a1.text_input("group id (short, no spaces)", key="new_cl_id", disabled=locked)
    new_label = a2.text_input("what it is", key="new_cl_label", disabled=locked)
    a3, a4 = st.columns(2)
    new_effort = a3.selectbox("how much work", EFFORT_BUCKETS, index=1, key="new_cl_effort", disabled=locked)
    new_conf = a4.slider("how sure", 0.0, 1.0, 0.5, 0.05, key="new_cl_conf", disabled=locked)
    new_hyp = st.text_input("what we think the shared cause is", key="new_cl_hyp", disabled=locked)
    if st.button("Add group", disabled=locked or not new_id.strip()):
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
        "cluster_id": st.column_config.TextColumn("group", width="small"),
        "label": st.column_config.TextColumn("what it is", width="medium"),
        "ticket_ids": st.column_config.TextColumn("tickets (model's suggestion)", width="small"),
        "effort_bucket": st.column_config.SelectboxColumn("how much work", options=EFFORT_BUCKETS, width="small"),
        "confidence": st.column_config.NumberColumn("how sure", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
        "root_cause_hypothesis": st.column_config.TextColumn("shared cause", width="large"),
        "evidence": st.column_config.TextColumn("evidence", width="large"),
    },
)
working_clusters = ss.clusters if locked else cl_editor
ss._pending_clusters = working_clusters
cluster_ids = [""] + sorted(working_clusters["cluster_id"].astype(str).tolist()) if len(working_clusters) else [""]

if not locked and len(working_clusters):
    rm1, rm2 = st.columns([1, 3])
    victim = rm1.selectbox("Remove a group", [""] + [c for c in cluster_ids if c], key="rm_cl")
    if rm2.button("Remove", disabled=not victim):
        ss.clusters = working_clusters[working_clusters["cluster_id"] != victim].reset_index(drop=True)
        base_t = ss.get("_pending_tickets", ss.tickets).copy()
        base_t.loc[base_t["cluster_id"] == victim, "cluster_id"] = ""
        ss.tickets = base_t
        _clear("cluster_editor", "ticket_editor", "rm_cl")
        st.rerun()

# ---- 3c tickets ----
st.subheader("3c. Tickets")
st.caption("Pick a group to put a ticket in one. For initiatives, list the request ids it would make go away, separated by commas.")
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
        "classification": st.column_config.SelectboxColumn("request or initiative", options=CLASSIFICATIONS, width="small"),
        "cluster_id": st.column_config.SelectboxColumn("group", options=cluster_ids, width="small"),
        "severity": st.column_config.SelectboxColumn("how serious", options=SEVERITIES, width="small"),
        "confidence": st.column_config.NumberColumn("how sure", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
        "effort_bucket": st.column_config.SelectboxColumn("how much work", options=EFFORT_BUCKETS, width="small"),
        "redirect": st.column_config.CheckboxColumn("hand off", width="small"),
        "metric_linked": st.column_config.CheckboxColumn("company goal", width="small"),
        "retires": st.column_config.TextColumn("makes go away", width="medium"),
        "reviewer_note": st.column_config.TextColumn("your note", width="medium"),
        "reason": st.column_config.TextColumn("model's reason", width="large"),
        "ambiguity_note": st.column_config.TextColumn("model flagged for you", width="large"),
        "summary": st.column_config.TextColumn("summary", width="large"),
    },
)
working_tickets = ss.tickets if locked else tk_editor
ss._pending_tickets = working_tickets

if len(working_clusters):
    counts = working_tickets["cluster_id"].fillna("").astype(str).value_counts()
    lonely = [c for c in working_clusters["cluster_id"].astype(str) if counts.get(c, 0) < 2]
    if lonely:
        st.caption(f"Groups with fewer than two tickets (scored as single tickets until you add more): {', '.join(lonely)}")
unknown = sorted(set(working_tickets["cluster_id"].fillna("").astype(str)) - set(cluster_ids))
if unknown:
    st.warning(f"Some tickets point at groups that no longer exist: {', '.join(unknown)}. They will score on their own values.")

if not locked and st.button("Throw away my edits and go back to the model's suggestions"):
    ss.tickets = ss.ai_tickets.copy()
    ss.clusters = ss.ai_clusters.copy()
    ss.accounts_df = build_accounts_df(meta["result"].get("accounts", []))
    _clear("ticket_editor", "cluster_editor", "account_editor")
    st.rerun()

# --------------------------------------------------------------------------- #
# Step 4: results
# --------------------------------------------------------------------------- #
st.header("Step 4. Results" + ("" if locked else "  (live; updates as you edit)"))

try:
    scored, clusters_scored = score_backlog(working_tickets, working_clusters, config)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

split = capacity_split(scored, config)
diff = diff_tags(ss.ai_tickets, working_tickets)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customer requests", f"{split['reactive_pct']}%", f"{split['reactive_points']:g} pts")
m2.metric("Our own initiatives", f"{split['proactive_pct']}%", f"{split['proactive_points']:g} pts")
m3.metric("Points used", f"{split['committed_points']:g} of {split['capacity_points']:g}", f"{split['headroom_points']:g} free")
m4.metric("Requests by ticket count", f"{split['reactive_in_backlog_pct']}%")
m5.metric("Suggestions you changed", len(diff), f"{diff['request_id'].nunique()} tickets" if len(diff) else None)

st.subheader("Why this split")
st.markdown(explain_split(scored, clusters_scored, split, config, working_tickets, working_clusters))

DISPLAY_COLS = {
    "request_id": "ticket", "source_account": "account", "classification": "type", "cluster_id": "group",
    "severity": "how serious", "eff_confidence": "how sure", "eff_effort_bucket": "work", "impact": "matters",
    "cost_share": "points", "score": "score", "capacity_note": "note", "reason": "why",
}


def show(sub: pd.DataFrame) -> None:
    st.dataframe(sub[list(DISPLAY_COLS)].rename(columns=DISPLAY_COLS), width="stretch", hide_index=True,
                 column_config={"why": st.column_config.TextColumn("why", width="large")})


tab_labels = BUCKET_ORDER + ["Groups", "What we assumed", "What you changed", "Everything"]
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
    st.caption("One row per fix. Single tickets appear as their own group.")
    st.dataframe(clusters_scored.rename(columns={"key": "group", "label": "what it is", "breadth": "tickets in group",
                                                 "effort_bucket": "work", "confidence": "how sure", "cluster_impact": "matters",
                                                 "cost_points": "points"}), width="stretch", hide_index=True)
with tabs[len(BUCKET_ORDER) + 1]:
    st.markdown(list_assumptions(scored, clusters_scored, config, meta["result"].get("accounts", []), meta, diff))
with tabs[len(BUCKET_ORDER) + 2]:
    if diff.empty:
        st.write("You have not changed any of the model's suggestions.")
    else:
        st.dataframe(diff.rename(columns={"ai_proposed": "model suggested", "reviewer_final": "you chose", "reviewer_note": "your note"}),
                     width="stretch", hide_index=True)
with tabs[len(BUCKET_ORDER) + 3]:
    st.dataframe(scored.drop(columns=["summary"], errors="ignore"), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Step 5: finalise
# --------------------------------------------------------------------------- #
st.header("Step 5. Finalise and download")
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
    st.caption("Finalise locks the tables and turns on the downloads.")
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
    e3.download_button("What you changed (CSV)", diff.to_csv(index=False), "changes_from_model.csv", "text/csv")
    e4.download_button("Final tags and settings (JSON)", json.dumps(final_json, indent=2), "finalized.json", "application/json")
    with st.expander("Preview the summary"):
        st.markdown(summary_md)
