"""Plain-language text that ships with the results: why this split, what we assumed, and the summary file."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from scoring import BUCKET_HELP, BUCKET_ORDER, capacity_split, score_backlog


def _pts(n: float) -> str:
    return f"{n:g} point{'s' if n != 1 else ''}"


def explain_split(scored: pd.DataFrame, clusters: pd.DataFrame, split: dict, config: dict,
                  tickets: pd.DataFrame | None = None, cluster_df: pd.DataFrame | None = None) -> str:
    """Why the split is what it is, built from what is actually in each pile."""
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
                out.append(f"one fix for {ids} ({_pts(r['cost_points'])}, score {r['score']})")
            else:
                out.append(f"{r['request_id']} ({_pts(r['cost_points'])}, score {r['score']})")
        return out

    n_fixes = len(fixes(do_r))
    lines = [
        f"**We recommend spending {split['reactive_pct']}% of next quarter on customer requests and "
        f"{split['proactive_pct']}% on our own initiatives.** "
        f"That is {_pts(split['reactive_points'])} on requests and {_pts(split['proactive_points'])} on initiatives, "
        f"{_pts(split['committed_points'])} in total, out of {_pts(split['capacity_points'])} the team can do. "
        f"{_pts(split['headroom_points'])} left unallocated.",
        "",
        f"If you just counted tickets, {split['reactive_in_backlog_pct']}% of the backlog is customer requests. "
        "The recommendation is lower than that for two reasons: we measure the split by how much work each item is, not how many tickets there are, "
        "and requests that share the same root cause are fixed once and paid for once.",
        "",
        f"**Customer requests we would do now ({len(do_r)} tickets, {n_fixes} pieces of work):** " + ("; ".join(fixes(do_r)) or "none") + ".",
    ]
    if len(inv):
        lines.append(f"We would also spend {_pts(inv['cost_share'].sum())} finding out what is behind {', '.join(inv['request_id'])}. "
                     "These matter enough to look into, but nobody knows the cause yet, so estimating them now would be a guess.")
    lines.append("")
    if len(do_p):
        bits = []
        for _, r in do_p.iterrows():
            ret = [x for x in r["retires"] if x] if isinstance(r["retires"], list) else []
            why = f"it would make {len(ret)} of the requests above go away for good ({', '.join(ret)})" if ret else "it stands on its own metric case"
            bits.append(f"{r['request_id']} ({_pts(r['cost_points'])}, score {r['score']}): {why}")
        lines.append(f"**Our own initiatives we would do now ({len(do_p)}):** " + "; ".join(bits) + ".")
        lines.append("An initiative earns its place by removing future requests. One that removes a whole group of requests is scored on all of them together, "
                     "which is how it can beat a single loud request.")
    else:
        lines.append("**Our own initiatives:** none make the cut at the current funding line. "
                     f"Every initiative scored below {config['do_now_threshold']:g}. Before accepting that, check the 'makes go away' list and the work size on each one.")
    lines.append("")
    if len(later):
        top = later.sort_values("score", ascending=False).iloc[0]
        fits = "would fit in the unallocated points" if top["cost_points"] <= split["headroom_points"] else "would not fit in the unallocated points"
        kind = "initiative" if top["classification"] == "proactive" else "request"
        lines.append(f"**First in line if something changes:** {top['request_id']} (an {kind}, score {top['score']}, {_pts(top['cost_points'])}) is the best of the 'Later' pile and {fits}. "
                     "If its work size or how sure we are turns out to be wrong, it is the first thing that moves.")
    if len(handoff):
        lines.append(f"**Handed off, not counted:** {len(handoff)} items ({', '.join(handoff['request_id'])}) are config changes, data cleanups or process work. "
                     "They should get done, some this week, but by support, infra or the PM group, not from engineering time. "
                     "Leaving them in the product backlog is part of why the request share looks so large.")
    if len(do_p) and tickets is not None and cluster_df is not None:
        low = float(do_p["score"].min())
        alt = dict(config); alt["do_now_threshold"] = round(low + 0.01, 2)
        try:
            alt_scored, _ = score_backlog(tickets, cluster_df, alt)
            alt_split = capacity_split(alt_scored, alt)
            lines.append("")
            lines.append(f"**How solid is this?** If the funding line moved from {config['do_now_threshold']:g} to {alt['do_now_threshold']:g}, "
                         f"the lowest-scoring initiative would drop out and the split would become {alt_split['reactive_pct']}/{alt_split['proactive_pct']}. "
                         "The initiative share depends on where that line sits. Anyone who disagrees can move it in the settings and see the result.")
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
    retire_note = "; ".join(f"{r['request_id']} removes {', '.join(r['retires']) or 'nothing'}" for _, r in pro.iterrows())

    lines = [
        "**About the accounts**",
        "- The data does not say how much revenue each account is worth. The AI model reads size clues in the notes (\"top-5 by GMV\", \"enterprise\", \"small regional\") "
        "and suggests an importance weight from 1 to 5. Weights in use: " + ", ".join(f"{k} {v:g}" for k, v in sorted(aw.items())) + ".",
        (f"- The reviewer changed the suggested weight for: {', '.join(changed_w)}." if changed_w else "- All weights are the model's suggestions; the reviewer has not changed any."),
        (f"- No size clue was found for {', '.join(unknown_tier)}, so they were given the middle weight (3)." if unknown_tier else "- Every account had a size clue in the notes."),
        "",
        "**About the tickets**",
        "- How serious a ticket is comes from the numbers and facts in the notes, not from how urgent the email sounds. Where the two disagree the notes win"
        + (f"; {len(flagged)} tickets carry a note about something the reviewer had to judge: {', '.join(flagged)}." if flagged else "."),
        f"- Work size turns into points: small {ep['small']:g}, 1-2 sprints {ep['1-2 sprints']:g}, large {ep['large']:g}. "
        f"When nobody knows the size, we do not guess; we spend {ep['unclear']:g} point finding out, if the ticket matters enough"
        + (f". Unknown size right now: {', '.join(unclear)}." if unclear else "."),
        "- \"Next quarter\" means the quarter after the latest date in the file.",
        "",
        "**About the groups (tickets that share one fix)**",
        "- A group is a guess that several tickets have the same cause and would be fixed together, so the work is counted once. Engineering has not confirmed any of them. "
        "If a group is wrong, we have given one fix too much credit and under-counted the work.",
        (f"- Groups we are less than 60% sure about: {', '.join(low_conf)}." if low_conf else "- We are at least 60% sure about every group."),
        "",
        "**About our own initiatives**",
        "- An initiative only gets credit for requests in this backlog that it would stop from happening again. Spotting a problem sooner is not the same as fixing it. "
        "Where the model suggested a spot-it-sooner link, the ticket's note says so and the reviewer decides.",
        f"- Current links: {retire_note or 'none'}.",
        f"- Initiatives tied to a measurable company goal get a {config['metric_bonus']:g}x bonus. A useful initiative with no named goal gets no bonus.",
        "",
        "**About capacity and the split**",
        f"- The team can do {config['quarter_capacity_points']:g} points this quarter. The default assumes five people, six two-week sprints, and about five points a sprint after support work. The data does not say how big the team is.",
        "- The split is measured in points of work (Do now plus Investigate first), not in ticket counts. Handed-off items are left out because config, cleanup and process work do not use engineering time, "
        "assuming support, infra or the PM group actually pick them up.",
        f"- The funding line ({config['do_now_threshold']:g}) is a chosen number, not something the data tells us. The 'How solid is this?' note above shows what moving it does.",
        "",
        "**About the AI pass**",
        "- The AI model reads the text and suggests tags. It does not score anything. Its reading can differ between runs, so each run is saved and reused.",
        (f"- This proposal: {meta.get('source')}, model {meta.get('model')}, generated {meta.get('generated_at')}." if meta else "- No proposal loaded."),
        (f"- The reviewer changed {len(diff)} field(s) on {diff['request_id'].nunique()} ticket(s). See 'What you changed'." if len(diff)
         else "- The reviewer has not changed any suggested tag yet. Everything in use is the model's reading."),
    ]
    return "\n".join(lines)


def build_summary(scored, clusters, split, config, diff, ai_meta, observations, tickets=None, cluster_df=None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    accounts = (ai_meta or {}).get("result", {}).get("accounts") if ai_meta else None
    lines = [
        "# Backlog scoring: customer requests vs our own initiatives",
        f"Generated {now}.",
        "",
        "## Recommended split",
        explain_split(scored, clusters, split, config, tickets, cluster_df),
        "",
        "## What we assumed",
        list_assumptions(scored, clusters, config, accounts, ai_meta, diff),
        "",
        "## How the score works",
        "score = how much it matters x how sure we are / how much work it is.",
        "How much it matters = account weight (1 to 5) x how serious (low 1, medium 2, high 3). "
        "Tickets that share one fix are scored together: their 'matters' points are added up, the work is counted once, and every ticket in the group gets the group's score.",
        "Our own initiatives get their own 'matters' points plus the points of every request they would make go away, times a bonus if tied to a measurable company goal.",
        "Piles: " + " ".join(f"{b}: {BUCKET_HELP[b]}" for b in BUCKET_ORDER),
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
            grp = f" [group: {r['cluster_id']}]" if r.get("cluster_id") else ""
            note = f" ({r['capacity_note']})" if r.get("capacity_note") else ""
            lines.append(f"- {r['request_id']}{grp} {r['source_account']} | {r['classification']} | score {r['score']} | {r['cost_share']} pts{note}: {r.get('reason', '')}")
        lines.append("")
    lines.append("## Groups")
    for _, c in clusters[~clusters["key"].str.startswith("single:")].iterrows():
        lines.append(f"- {c['key']}: {c['label']} ({c['tickets']}); work {c['effort_bucket']}, sure {c['confidence']}, matters {c['cluster_impact']}, score {c['score']}")
    lines.append("")
    if observations:
        lines.append("## What the AI model noticed across tickets")
        lines.extend(f"- {o}" for o in observations)
        lines.append("")
    lines.append("## What the reviewer changed")
    if diff.empty:
        lines.append("Nothing changed from the AI proposal.")
    else:
        lines.append(f"{len(diff)} field(s) on {diff['request_id'].nunique()} ticket(s).")
        for _, d in diff.iterrows():
            note = f" ({d['reviewer_note']})" if d.get("reviewer_note") else ""
            lines.append(f"- {d['request_id']} {d['field']}: {d['ai_proposed']} -> {d['reviewer_final']}{note}")
    lines.append("")
    return "\n".join(lines)
