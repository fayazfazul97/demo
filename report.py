"""Builds the markdown summary that ships with the scored CSV."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from scoring import BUCKET_ORDER


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
