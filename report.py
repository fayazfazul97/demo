"""Plain-language text that ships with the results: why this split, what we assumed, and the summary file."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from scoring import BUCKET_HELP, BUCKET_ORDER, capacity_split, score_backlog


def _pts(n: float) -> str:
    return f"{n:g} point{'s' if n != 1 else ''}"


def explain_split(scored: pd.DataFrame, clusters: pd.DataFrame, split: dict, config: dict,
                  tickets: pd.DataFrame | None = None, cluster_df: pd.DataFrame | None = None) -> str:
    """Rationale for the split, built from what is actually in each category."""
    do = scored[scored["bucket"] == "Do now"]
    inv = scored[scored["bucket"] == "Investigate first"]
    later = scored[scored["bucket"] == "Later"]
    handoff = scored[scored["bucket"] == "Hand off"]
    do_r = do[do["classification"] == "reactive"]
    do_p = do[do["classification"] == "proactive"]

    def fixes(sub: pd.DataFrame) -> list[str]:
        out, seen = [], set()
        for _, r in sub.sort_values("score", ascending=False).iterrows():
            key = r["cluster_id"] or r["request_id"]
            if key in seen:
                continue
            seen.add(key)
            if r["cluster_id"]:
                ids = ", ".join(sub[sub["cluster_id"] == r["cluster_id"]]["request_id"])
                out.append(f"one fix covering {ids} ({_pts(r['cost_points'])}, score {r['score']})")
            else:
                out.append(f"{r['request_id']} ({_pts(r['cost_points'])}, score {r['score']})")
        return out

    n_fixes = len(fixes(do_r))
    lines = [
        f"**Recommended allocation for next quarter: {split['reactive_pct']}% reactive (customer requests), "
        f"{split['proactive_pct']}% proactive (team initiatives).** "
        f"That is {_pts(split['reactive_points'])} of reactive work and {_pts(split['proactive_points'])} of proactive work, "
        f"{_pts(split['committed_points'])} in total, against a capacity of {_pts(split['capacity_points'])}. "
        f"{_pts(split['headroom_points'])} remain unallocated.",
        "",
        f"By ticket count, {split['reactive_in_backlog_pct']}% of the backlog is reactive. "
        "The recommendation differs from that figure for two reasons: the split is measured by the effort of each item, not by ticket count, "
        "and reactive tickets that share a root cause are fixed once and costed once.",
        "",
        f"**Reactive work prioritised ({len(do_r)} tickets, {n_fixes} fixes):** " + ("; ".join(fixes(do_r)) or "none") + ".",
    ]
    if len(inv):
        lines.append(f"A further {_pts(inv['cost_share'].sum())} are allocated to investigating {', '.join(inv['request_id'])}. "
                     "These have enough impact to warrant a look, but the cause is not yet known, so an estimate now would be a guess.")
    lines.append("")
    if len(do_p):
        bits = []
        for _, r in do_p.iterrows():
            ret = [x for x in r["retires"] if x] if isinstance(r["retires"], list) else []
            why = f"it would eliminate {len(ret)} of the reactive tickets above ({', '.join(ret)})" if ret else "it is justified on its metric case alone"
            bits.append(f"{r['request_id']} ({_pts(r['cost_points'])}, score {r['score']}): {why}")
        lines.append(f"**Proactive work prioritised ({len(do_p)}):** " + "; ".join(bits) + ".")
        lines.append("A proactive item earns its place by eliminating future reactive work. One that eliminates a whole cluster is scored on the combined impact of that cluster, "
                     "which is how it can outrank a single loud request.")
    else:
        lines.append("**Proactive work:** none clears the current priority threshold. "
                     f"Every proactive item scored below {config['do_now_threshold']:g}. Before accepting that, check the 'eliminates' list and the effort on each one.")
    lines.append("")
    pulled = scored[scored["capacity_note"].astype(str).str.startswith("pulled up")]
    if len(pulled):
        items = "; ".join(f"{r['request_id']} ({r['classification']}, score {r['score']}, {_pts(r['cost_points'])})" for _, r in pulled.iterrows())
        lines.append(f"**Filled from lower categories:** {len(pulled)} item(s) scored below the threshold but were pulled into 'Do now' because capacity was available: {items}. "
                     "They are funded on spare capacity, not on merit against the threshold; if capacity tightens they are the first to drop.")
    if len(later):
        top = later.sort_values("score", ascending=False).iloc[0]
        fits = "would fit within the unallocated points" if top["cost_points"] <= split["headroom_points"] else "would not fit within the unallocated points"
        kind = "proactive item" if top["classification"] == "proactive" else "reactive ticket"
        lines.append(f"**Next in line:** {top['request_id']} (a {kind}, score {top['score']}, {_pts(top['cost_points'])}) is the highest-scoring item in 'Later' and {fits}. "
                     "If its effort or confidence is revised, it is the first item to move.")
    if len(handoff):
        lines.append(f"**Handed off, excluded from capacity:** {len(handoff)} items ({', '.join(handoff['request_id'])}) are configuration changes, data cleanups or process work. "
                     "They should be completed, some within the week, but by support, infrastructure or the PM group rather than from engineering capacity. "
                     "Leaving them in the product backlog inflates the apparent reactive share.")
    if len(do_p) and tickets is not None and cluster_df is not None:
        low = float(do_p["score"].min())
        alt = dict(config); alt["do_now_threshold"] = round(low + 0.01, 2)
        try:
            alt_scored, _ = score_backlog(tickets, cluster_df, alt)
            alt_split = capacity_split(alt_scored, alt)
            lines.append("")
            lines.append(f"**Sensitivity:** if the priority threshold moved from {config['do_now_threshold']:g} to {alt['do_now_threshold']:g}, "
                         f"the lowest-scoring proactive item would drop out and the split would become {alt_split['reactive_pct']}/{alt_split['proactive_pct']}. "
                         "The proactive share depends on where that threshold sits. The setting can be adjusted in the sidebar to test alternatives.")
        except Exception:
            pass
    return "\n".join(lines)


def list_assumptions(scored: pd.DataFrame, clusters: pd.DataFrame, config: dict,
                     accounts: list[dict] | None, meta: dict | None, diff: pd.DataFrame) -> str:
    aw = config["account_weight"]
    ep = config["effort_points"]
    accounts = accounts or []
    unknown_tier = [a["source_account"] for a in accounts if a.get("tier") == "unknown"]
    changed_w = [a["source_account"] for a in accounts
                 if a["source_account"] in aw and float(aw[a["source_account"]]) != float(a.get("suggested_weight", aw[a["source_account"]]))]
    low_conf = clusters[(~clusters["key"].str.startswith("single:")) & (clusters["confidence"] < 0.6)]["key"].tolist() if len(clusters) else []
    unclear = scored[scored["eff_effort_bucket"] == "unclear"]["request_id"].tolist()
    flagged = scored[scored["ambiguity_note"].fillna("").astype(str).str.strip() != ""]["request_id"].tolist() if "ambiguity_note" in scored else []
    pro = scored[scored["classification"] == "proactive"]
    retire_note = "; ".join(f"{r['request_id']} eliminates {', '.join(r['retires']) or 'nothing'}" for _, r in pro.iterrows())

    lines = [
        "**About the accounts**",
        "- The data does not state each account's revenue. The AI model reads size indicators in the notes (\"top-5 by GMV\", \"enterprise\", \"small regional\") "
        "and proposes an account size from 1 to 5. Sizes in use: " + ", ".join(f"{k} {v:g}" for k, v in sorted(aw.items())) + ".",
        (f"- The reviewer changed the proposed account size for: {', '.join(changed_w)}." if changed_w else "- All account sizes are the model's proposals; the reviewer has not changed any."),
        (f"- No size indicator was found for {', '.join(unknown_tier)}, so they were assigned the middle size (3)." if unknown_tier else "- Every account had a size indicator in the notes."),
        "",
        "**About the tickets**",
        "- Severity is based on the figures and facts in the notes, not on the urgency of the tone. Where the two disagree, the notes prevail"
        + (f"; {len(flagged)} tickets carry a note flagging a judgement for the reviewer: {', '.join(flagged)}." if flagged else "."),
        f"- Effort turns into points: small {ep['small']:g}, 1-2 sprints {ep['1-2 sprints']:g}, large {ep['large']:g}. "
        f"When effort is unclear it is not estimated; {ep['unclear']:g} point is allocated to a time-boxed investigation, provided the impact justifies it"
        + (f". Effort unclear right now: {', '.join(unclear)}." if unclear else "."),
        "- \"Next quarter\" refers to the quarter following the latest date_received in the file.",
        "",
        "**About the clusters (tickets that share one root cause)**",
        "- A cluster is a hypothesis that several tickets share one root cause and one fix, so the effort is counted once. Engineering has not confirmed any of them. "
        "If a cluster is wrong, one fix has been over-credited and the effort under-counted.",
        (f"- Clusters with confidence below 0.6, to be treated as hypotheses: {', '.join(low_conf)}." if low_conf else "- Every cluster has confidence of 0.6 or above."),
        "",
        "**About proactive items**",
        "- A proactive item is credited only for reactive tickets in this backlog that it would eliminate or stop from recurring. Detecting a problem sooner is not the same as fixing it. "
        "Where the model proposed a detection-only link, the ticket's note says so and the reviewer decides.",
        f"- Current links: {retire_note or 'none'}.",
        f"- Proactive items linked to a named company metric receive a {config['metric_bonus']:g}x bonus. A useful item with no named metric receives no bonus.",
        "",
        "**About capacity and the split**",
        f"- Team capacity is {config['quarter_capacity_points']:g} points this quarter. The default assumes five people, six two-week sprints, and roughly five points per sprint after support load. The data does not state team size.",
        "- The split is measured in points of effort (Do now plus Investigate first), not in ticket counts. Handed-off items are excluded because configuration, cleanup and process work do not consume engineering capacity, "
        "on the assumption that support, infrastructure or the PM group take them on.",
        f"- The priority threshold ({config['do_now_threshold']:g}) is a chosen value, not derived from the data. The sensitivity note in the rationale shows the effect of moving it.",
        {"later": "- Spare capacity is filled from 'Later' with the best-scoring items that fit, so the quarter is fully allocated. Items funded this way are marked 'pulled up' and are the first to drop if capacity tightens.",
         "later_and_declined": "- Spare capacity is filled from 'Later' and 'Not this quarter' with the best-scoring items that fit. Items funded this way are marked 'pulled up' and are the first to drop if capacity tightens.",
         "off": "- Spare capacity is left unallocated; nothing below the threshold is funded."}[config.get("fill_spare_capacity", "later")],
        "",
        "**About the AI pass**",
        "- The AI model reads the text and proposes tags. It does not score. Its reading can vary between runs, so each run is saved and reused.",
        (f"- This proposal: {meta.get('source')}, model {meta.get('model')}, generated {meta.get('generated_at')}." if meta else "- No proposal loaded."),
        (f"- The reviewer changed {len(diff)} field(s) on {diff['request_id'].nunique()} ticket(s). See 'Changes from the model'." if len(diff)
         else "- The reviewer has not changed any proposed tag. Every tag in use is the model's reading."),
    ]
    return "\n".join(lines)


def build_summary(scored, clusters, split, config, diff, ai_meta, observations, tickets=None, cluster_df=None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    accounts = (ai_meta or {}).get("result", {}).get("accounts") if ai_meta else None
    lines = [
        "# Backlog scoring: reactive vs proactive work",
        f"Generated {now}.",
        "",
        "## Recommended allocation",
        explain_split(scored, clusters, split, config, tickets, cluster_df),
        "",
        "## Assumptions",
        list_assumptions(scored, clusters, config, accounts, ai_meta, diff),
        "",
        "## How the score works",
        "score = impact x confidence / effort.",
        "Impact = account size (1 to 5) x severity (low 1, medium 2, high 3). "
        "Tickets that share one root cause are scored as a cluster: their impact is summed, the effort is counted once, and every ticket in the cluster receives the cluster's score.",
        "Proactive items receive their own impact plus the impact of every reactive ticket they would eliminate, multiplied by a bonus if linked to a company metric.",
        "Categories: " + " ".join(f"{b}: {BUCKET_HELP[b]}" for b in BUCKET_ORDER),
        "",
        "## Settings used",
    ]
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    for bucket in BUCKET_ORDER:
        sub = scored[scored["bucket"] == bucket]
        if sub.empty:
            continue
        lines.append(f"## {bucket} ({len(sub)})")
        for _, r in sub.iterrows():
            grp = f" [cluster: {r['cluster_id']}]" if r.get("cluster_id") else ""
            note = f" ({r['capacity_note']})" if r.get("capacity_note") else ""
            lines.append(f"- {r['request_id']}{grp} {r['source_account']} | {r['classification']} | score {r['score']} | {r['cost_share']} pts{note}: {r.get('reason', '')}")
        lines.append("")
    lines.append("## Clusters")
    for _, c in clusters[~clusters["key"].str.startswith("single:")].iterrows():
        lines.append(f"- {c['key']}: {c['label']} ({c['tickets']}); effort {c['effort_bucket']}, confidence {c['confidence']}, impact {c['cluster_impact']}, score {c['score']}")
    lines.append("")
    if observations:
        lines.append("## Cross-ticket observations from the model")
        lines.extend(f"- {o}" for o in observations)
        lines.append("")
    lines.append("## Changes from the model")
    if diff.empty:
        lines.append("No changes from the model's proposals.")
    else:
        lines.append(f"{len(diff)} field(s) on {diff['request_id'].nunique()} ticket(s).")
        for _, d in diff.iterrows():
            note = f" ({d['reviewer_note']})" if d.get("reviewer_note") else ""
            lines.append(f"- {d['request_id']} {d['field']}: {d['ai_proposed']} -> {d['reviewer_final']}{note}")
    lines.append("")
    return "\n".join(lines)
