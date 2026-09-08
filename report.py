"""Builds the markdown summary that ships with the scored CSV."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from scoring import BUCKET_ORDER, TIER_WEIGHTS, capacity_split, score_backlog


def explain_split(scored: pd.DataFrame, clusters: pd.DataFrame, split: dict, config: dict,
                  tickets: pd.DataFrame | None = None, cluster_df: pd.DataFrame | None = None) -> str:
    """Plain-language reasoning for the split, built from what is actually in the buckets."""
    do = scored[scored["bucket"] == "Do now"]
    spikes = scored[scored["bucket"] == "Discovery spike"]
    defer = scored[scored["bucket"] == "Defer"]
    redirect = scored[scored["bucket"] == "Redirect"]
    do_r = do[do["classification"] == "reactive"]
    do_p = do[do["classification"] == "proactive"]

    def fix_list(sub: pd.DataFrame) -> str:
        parts = []
        seen = set()
        for _, r in sub.sort_values("score", ascending=False).iterrows():
            key = r["cluster_id"] if r["cluster_id"] else r["request_id"]
            if key in seen:
                continue
            seen.add(key)
            if r["cluster_id"]:
                ids = ", ".join(sub[sub["cluster_id"] == r["cluster_id"]]["request_id"])
                parts.append(f"{r['cluster_id']} ({ids}; {r['cost_points']:g} pt{'s' if r['cost_points'] != 1 else ''}, score {r['score']})")
            else:
                parts.append(f"{r['request_id']} ({r['cost_points']:g} pt{'s' if r['cost_points'] != 1 else ''}, score {r['score']})")
        return "; ".join(parts) if parts else "nothing"

    lines = [
        f"**Recommended split: {split['reactive_pct']}% reactive / {split['proactive_pct']}% proactive.** "
        f"That is {split['reactive_points']:g} reactive and {split['proactive_points']:g} proactive points out of "
        f"{split['committed_points']:g} committed, against {split['capacity_points']:g} points of capacity "
        f"({split['headroom_points']:g} points unallocated).",
        "",
        f"By ticket count the backlog is {split['reactive_in_backlog_pct']}% reactive. The recommendation is not that number, "
        "because the split is measured in cost of the work that clears the funding bar, not in tickets, and because "
        "reactive tickets that share a root cause are paid for once.",
        "",
        f"**Reactive side ({len(do_r)} tickets funded as {do_r['cluster_id'].replace('', pd.NA).nunique() + (do_r['cluster_id'] == '').sum()} fixes):** {fix_list(do_r)}.",
    ]
    if len(spikes):
        lines.append(f"Plus {len(spikes)} discovery spike(s) at {spikes['cost_share'].sum():g} pts: "
                     f"{', '.join(spikes['request_id'])}. These are unknowns with enough impact to be worth a time-boxed look before anyone estimates them.")
    lines.append("")
    if len(do_p):
        bits = []
        for _, r in do_p.iterrows():
            ret = [x for x in r["retires"] if x] if isinstance(r["retires"], list) else []
            why = f"retires {len(ret)} reactive ticket(s): {', '.join(ret)}" if ret else "no reactive tickets retired; funded on its own metric case"
            bits.append(f"{r['request_id']} ({r['cost_points']:g} pts, score {r['score']}; {why})")
        lines.append(f"**Proactive side ({len(do_p)} item(s)):** " + "; ".join(bits) + ".")
        lines.append("Proactive work earns its share by removing future reactive work. An item that retires a cluster competes "
                     "on the combined impact of that cluster, which is why it can outrank any single loud request.")
    else:
        lines.append("**Proactive side:** nothing clears the bar at the current threshold. Every proactive item scored below "
                     f"{config['do_now_threshold']:g}; check whether their `retires` lists and effort buckets are right before accepting that.")
    lines.append("")
    if len(defer):
        top = defer.sort_values("score", ascending=False).iloc[0]
        fits = "fits in the headroom" if top["cost_points"] <= split["headroom_points"] else "does not fit in the headroom"
        lines.append(f"**Next in line:** {top['request_id']} ({top['classification']}, score {top['score']}, {top['cost_points']:g} pts) is the highest-scoring deferred item and {fits}. "
                     f"If the effort or confidence on it is wrong, it is the first thing that moves.")
    if len(redirect):
        lines.append(f"**Routed off the roadmap:** {len(redirect)} item(s) ({', '.join(redirect['request_id'])}) are configuration, data cleanup, or process work. "
                     "They should be done, some of them this week, but by support, infra, or the PM group, not out of engineering capacity. "
                     "Leaving them in the product backlog is what makes the reactive share look bigger than it is.")
    # sensitivity: what threshold would drop the lowest funded proactive item
    if len(do_p) and tickets is not None and cluster_df is not None:
        low = float(do_p["score"].min())
        alt = dict(config); alt["do_now_threshold"] = round(low + 0.01, 2)
        try:
            alt_scored, _ = score_backlog(tickets, cluster_df, alt)
            alt_split = capacity_split(alt_scored, alt)
            lines.append("")
            lines.append(f"**Sensitivity:** raising the do-now threshold from {config['do_now_threshold']:g} to {alt['do_now_threshold']:g} "
                         f"drops the lowest-scoring funded proactive item and moves the split to {alt_split['reactive_pct']}/{alt_split['proactive_pct']}. "
                         "The proactive share is a judgement about that line, and the sidebar lets a sceptic move it.")
        except Exception:
            pass
    return "\n".join(lines)


def list_assumptions(scored: pd.DataFrame, clusters: pd.DataFrame, config: dict,
                     tiers: list[dict] | None, meta: dict | None, diff: pd.DataFrame) -> str:
    aw = config["account_weight"]
    ep = config["effort_points"]
    tiers = tiers or []
    unknown_tier = [t["source_account"] for t in tiers if t.get("tier") == "unknown"]
    low_conf_clusters = clusters[(~clusters["key"].str.startswith("single:")) & (clusters["confidence"] < 0.6)]["key"].tolist() if len(clusters) else []
    unclear = scored[scored["eff_effort_bucket"] == "unclear"]["request_id"].tolist()
    flagged = scored[scored["ambiguity_note"].fillna("").astype(str).str.strip() != ""]["request_id"].tolist() if "ambiguity_note" in scored else []
    proactive_ret = scored[(scored["classification"] == "proactive")]
    retire_note = ", ".join(f"{r['request_id']} -> {', '.join(r['retires']) or 'none'}" for _, r in proactive_ret.iterrows())
    latest = scored["date_received"].iloc[0] if "date_received" in scored and len(scored) else None

    lines = [
        "**About the data**",
        "- GMV and revenue per account are not in the dataset. Account weight comes from tier language in the notes, read by the AI model and mapped as "
        + ", ".join(f"{k} {v}" for k, v in TIER_WEIGHTS.items()) + ". Current weights: "
        + ", ".join(f"{k} {v:g}" for k, v in sorted(aw.items())) + ".",
        (f"- Accounts with no tier signal in the notes were weighted as unknown ({TIER_WEIGHTS['unknown']}): {', '.join(unknown_tier)}." if unknown_tier
         else "- Every account had a tier signal in the notes; none defaulted to unknown."),
        "- Severity is taken from the measured or described effect in raw_notes, not from the tone of the summary or subject line. "
        "Where the two disagree the notes win and the disagreement is recorded in the ambiguity note"
        + (f" ({len(flagged)} tickets flagged: {', '.join(flagged)})." if flagged else "."),
        f"- Effort hints map to points: small {ep['small']:g}, 1-2 sprints {ep['1-2 sprints']:g}, large {ep['large']:g}. "
        f"'Unclear' is never estimated; it is a {ep['unclear']:g}-point discovery spike when impact justifies one, otherwise scored on what is known"
        + (f". Tickets with unclear effort: {', '.join(unclear)}." if unclear else "."),
        "- 'Next quarter' means the quarter after the latest date_received in the file.",
        "",
        "**About the clusters**",
        "- A cluster is a hypothesis that several tickets share one root cause and one fix, so the fix is paid for once. Engineering has not confirmed any of them; "
        "a wrong cluster over-credits one fix and under-costs the work.",
        (f"- Clusters held at confidence below 0.6, treat as hypotheses: {', '.join(low_conf_clusters)}." if low_conf_clusters
         else "- All clusters are at confidence 0.6 or above."),
        "- Cluster effort and confidence override the member tickets' own values.",
        "",
        "**About proactive items**",
        "- A proactive item is credited only for reactive tickets in this backlog that it would structurally eliminate or stop recurring. Detecting a problem earlier is not the same as fixing it; "
        "where the AI model proposed a detection-only link the ambiguity note says so and the reviewer decides.",
        f"- Current retire links: {retire_note or 'none'}.",
        f"- The metric-linked bonus ({config['metric_bonus']:g}x) applies only where the notes name a company metric. A structural fix with no named metric gets no bonus even if it is obviously useful.",
        "",
        "**About capacity and the split**",
        f"- Quarter capacity is {config['quarter_capacity_points']:g} points. The default assumes a five-person team, six two-week sprints, and roughly five points per sprint left after support load. Nothing in the data states team size.",
        "- The split counts committed points (Do now plus Discovery spikes), not ticket counts. Redirected items are excluded because config, data cleanup and process work do not consume engineering capacity, "
        "on the assumption that support, infra or the PM group actually pick them up.",
        f"- The do-now threshold ({config['do_now_threshold']:g}) is a chosen line, not derived from the data. The sensitivity note in the split explanation shows what moving it does.",
        "",
        "**About the AI pass**",
        "- The AI model reads text and proposes tags; it does not score. Its output is a reading of the notes, can differ between runs, and is cached so the demo is stable.",
        (f"- Proposal source: {meta.get('source')}, model: {meta.get('model')}, generated {meta.get('generated_at')}." if meta else "- No proposal loaded."),
        (f"- The reviewer changed {len(diff)} field(s) on {diff['request_id'].nunique()} ticket(s) from the AI proposal; see the diff log." if len(diff)
         else "- The reviewer has not changed any AI-proposed field yet. Every tag currently in use is the model's reading."),
    ]
    return "\n".join(lines)


def build_summary(
    scored: pd.DataFrame,
    clusters: pd.DataFrame,
    split: dict,
    config: dict,
    diff: pd.DataFrame,
    ai_meta: dict | None,
    observations: list[str],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Trellis backlog: reactive vs proactive scoring",
        f"Generated {now}.",
        "",
        "## Recommended split for next quarter",
        f"Reactive {split['reactive_pct']}% / proactive {split['proactive_pct']}% of committed capacity "
        f"({split['reactive_points']} + {split['proactive_points']} = {split['committed_points']} of "
        f"{split['capacity_points']} points; {split['headroom_points']} points headroom).",
        f"For comparison, {split['reactive_in_backlog_pct']}% of the raw backlog is reactive by ticket count.",
        "",
        "## Why this split",
        explain_split(scored, clusters, split, config),
        "",
        "## Assumptions",
        list_assumptions(scored, clusters, config, (ai_meta or {}).get("result", {}).get("accounts") if ai_meta else None, ai_meta, diff),
        "",
        "## Scoring formula",
        "Reactive tickets are scored at the root-cause cluster level: "
        "score = sum(account_weight x severity over the cluster) x confidence / effort_points. "
        "Every ticket in a cluster inherits the cluster score and pays an equal share of its cost.",
        "Proactive items: score = (own impact + impact of the reactive tickets they retire) x metric_bonus x confidence / effort_points.",
        "Buckets: Redirect (not product work) -> Discovery spike (unknown cause, enough impact) -> Do now (score >= threshold) -> Defer -> Decline. "
        "Do-now items spill to Defer when committed points exceed capacity.",
        "",
        "## Config used",
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
            cl = f" [{r['cluster_id']}]" if r.get("cluster_id") else ""
            note = f" ({r['capacity_note']})" if r.get("capacity_note") else ""
            lines.append(
                f"- {r['request_id']}{cl} {r['source_account']} | {r['classification']} | "
                f"score {r['score']} | cost share {r['cost_share']}{note}: {r.get('reason', '')}"
            )
        lines.append("")

    lines.append("## Root-cause clusters")
    for _, c in clusters[~clusters["key"].str.startswith("single:")].iterrows():
        lines.append(
            f"- {c['key']}: {c['label']} ({c['tickets']}); effort {c['effort_bucket']}, "
            f"confidence {c['confidence']}, cluster impact {c['cluster_impact']}, score {c['score']}"
        )
    lines.append("")

    if observations:
        lines.append("## Cross-record observations from the AI pass")
        lines.extend(f"- {o}" for o in observations)
        lines.append("")

    lines.append("## Reviewer changes to AI proposals")
    if ai_meta:
        lines.append(f"AI pass source: {ai_meta.get('source')} | model: {ai_meta.get('model')} | generated: {ai_meta.get('generated_at')}")
    if diff.empty:
        lines.append("No fields changed from the AI proposal.")
    else:
        lines.append(f"{len(diff)} field(s) changed across {diff['request_id'].nunique()} ticket(s).")
        for _, d in diff.iterrows():
            note = f" ({d['reviewer_note']})" if d.get("reviewer_note") else ""
            lines.append(f"- {d['request_id']} {d['field']}: {d['ai_proposed']} -> {d['reviewer_final']}{note}")
    lines.append("")
    return "\n".join(lines)
