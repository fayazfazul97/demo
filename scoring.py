"""
Deterministic scoring layer.

Everything here is arithmetic on tags a human has finalised. No judgement lives
in this file; every number a stakeholder could argue with is in DEFAULT_CONFIG
and is exposed as a control in the app.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: dict[str, Any] = {
    # Relative weight per account. Populated at runtime from the accounts in
    # the uploaded file: the AI pass reads each account's tier from the notes,
    # TIER_WEIGHTS maps tier -> weight, and the reviewer can override any of
    # them in the sidebar. GMV is never in the data, so this is an assumption.
    "account_weight": {"Internal": 3},
    # Effort hint -> story points. "unclear" gets a discovery spike, not a guess.
    "effort_points": {"small": 1, "1-2 sprints": 3, "large": 8, "unclear": 1},
    "severity_scale": {"low": 1, "medium": 2, "high": 3},
    # Proactive items that name a company metric get this multiplier.
    "metric_bonus": 1.5,
    # Score thresholds. Defer band is [threshold * defer_ratio, threshold).
    "do_now_threshold": 3.0,
    "defer_ratio": 0.5,
    # Discovery spikes are only opened when confidence is below this AND the
    # (cluster) impact is at least discovery_min_impact. Otherwise an unknown
    # cause on a small account just gets scored on what we know.
    "discovery_confidence_max": 0.5,
    "discovery_min_impact": 3,
    # Story points the team can commit next quarter (assumption: 5 people,
    # 6 two-week sprints, ~5 points per sprint after support load).
    "quarter_capacity_points": 30,
}

TIER_WEIGHTS = {"top": 5, "enterprise": 5, "mid": 3, "small": 1, "internal": 3, "unknown": 3}

EFFORT_BUCKETS = ["small", "1-2 sprints", "large", "unclear"]
SEVERITIES = ["low", "medium", "high"]
CLASSIFICATIONS = ["reactive", "proactive"]
BUCKET_ORDER = ["Do now", "Investigate first", "Later", "Not this quarter", "Hand off"]
BUCKET_HELP = {
    "Do now": "Scores above the funding line and fits in this quarter's capacity.",
    "Investigate first": "We do not know the cause yet, but it matters enough to spend a little time finding out before estimating.",
    "Later": "Close to the line. Revisit if capacity frees up or the facts change.",
    "Not this quarter": "Scores well below the line. Say no for now and tell the account why.",
    "Hand off": "Not engineering work: a config change, a data cleanup, or a process. Route it to whoever owns that.",
}


# --------------------------------------------------------------------------- #
# Ingest helpers
# --------------------------------------------------------------------------- #

def read_backlog(path_or_buffer) -> pd.DataFrame:
    """The companion CSV is cp1252, not UTF-8. Try UTF-8 first, fall back."""
    try:
        df = pd.read_csv(path_or_buffer, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        df = pd.read_csv(path_or_buffer, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    df["effort_bucket"] = df["rough_effort_hint"].map(normalise_effort)
    return df


def normalise_effort(hint: str) -> str:
    h = (hint or "").lower()
    if h.startswith("unclear"):
        return "unclear"
    if h.startswith("large"):
        return "large"
    if h.startswith("1-2"):
        return "1-2 sprints"
    if h.startswith("small"):
        return "small"
    return "unclear"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_backlog(
    tickets: pd.DataFrame,
    clusters: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    tickets columns (finalised by the reviewer):
        request_id, source_account, classification, cluster_id, severity,
        confidence, effort_bucket, redirect, metric_linked, retires (list)
    clusters columns:
        cluster_id, label, effort_bucket, confidence

    Returns (scored_tickets, scored_clusters).
    """
    cfg = config
    aw = cfg["account_weight"]
    sev = cfg["severity_scale"]
    ep = cfg["effort_points"]

    t = tickets.copy()
    t["cluster_id"] = t["cluster_id"].fillna("").astype(str).str.strip()
    t["retires"] = t["retires"].apply(_as_list)
    t["confidence"] = pd.to_numeric(t["confidence"], errors="coerce").fillna(0.5).clip(0, 1)
    t["redirect"] = t["redirect"].fillna(False).astype(bool)
    t["metric_linked"] = t["metric_linked"].fillna(False).astype(bool)

    # Per-ticket raw impact: account weight x severity. Internal weight for proactive.
    def raw_impact(row):
        acct = "Internal" if row["classification"] == "proactive" else row["source_account"]
        return aw.get(acct, aw["Internal"]) * sev.get(row["severity"], 2)

    t["raw_impact"] = t.apply(raw_impact, axis=1)

    cl = clusters.copy() if clusters is not None and len(clusters) else pd.DataFrame(
        columns=["cluster_id", "label", "effort_bucket", "confidence"]
    )
    cl["cluster_id"] = cl["cluster_id"].astype(str).str.strip()
    cl = cl.set_index("cluster_id", drop=False)

    # ---- Reactive: score at the cluster level, tickets inherit ---------------
    impact_by_id = dict(zip(t["request_id"], t["raw_impact"]))
    reactive = t[t["classification"] == "reactive"]
    groups: dict[str, list[str]] = {}
    for _, r in reactive.iterrows():
        key = r["cluster_id"] if r["cluster_id"] else f"single:{r['request_id']}"
        groups.setdefault(key, []).append(r["request_id"])

    cluster_rows = []
    for key, ids in groups.items():
        members = t[t["request_id"].isin(ids)]
        if key.startswith("single:"):
            effort_bucket = members.iloc[0]["effort_bucket"]
            conf = float(members.iloc[0]["confidence"])
            label = members.iloc[0].get("reason", "") or ""
        else:
            if key in cl.index:
                effort_bucket = cl.loc[key, "effort_bucket"]
                conf = float(cl.loc[key, "confidence"])
                label = cl.loc[key, "label"]
            else:
                effort_bucket = members.iloc[0]["effort_bucket"]
                conf = float(members["confidence"].mean())
                label = key
        breadth = len(ids)
        cluster_impact = float(members["raw_impact"].sum())
        cost = float(ep.get(effort_bucket, ep["unclear"]))
        score = cluster_impact * conf / cost
        cluster_rows.append(
            dict(
                key=key,
                label=label,
                tickets=", ".join(ids),
                breadth=breadth,
                effort_bucket=effort_bucket,
                confidence=round(conf, 2),
                cluster_impact=round(cluster_impact, 2),
                cost_points=cost,
                score=round(score, 2),
            )
        )
        for rid in ids:
            idx = t.index[t["request_id"] == rid][0]
            t.loc[idx, "breadth"] = breadth
            t.loc[idx, "impact"] = cluster_impact
            t.loc[idx, "eff_confidence"] = conf
            t.loc[idx, "cost_points"] = cost
            t.loc[idx, "cost_share"] = cost / breadth
            t.loc[idx, "eff_effort_bucket"] = effort_bucket
            t.loc[idx, "score"] = score

    # ---- Proactive: own impact plus the reactive impact it retires ----------
    for idx, r in t[t["classification"] == "proactive"].iterrows():
        retired = [rid for rid in r["retires"] if rid in impact_by_id]
        retired_impact = sum(impact_by_id[rid] for rid in retired)
        impact = r["raw_impact"] + retired_impact
        if r["metric_linked"]:
            impact *= cfg["metric_bonus"]
        cost = float(ep.get(r["effort_bucket"], ep["unclear"]))
        conf = float(r["confidence"])
        t.loc[idx, "breadth"] = max(1, len(retired))
        t.loc[idx, "impact"] = impact
        t.loc[idx, "eff_confidence"] = conf
        t.loc[idx, "cost_points"] = cost
        t.loc[idx, "cost_share"] = cost
        t.loc[idx, "eff_effort_bucket"] = r["effort_bucket"]
        t.loc[idx, "score"] = impact * conf / cost

    # ---- Buckets --------------------------------------------------------------
    t["bucket"] = t.apply(lambda r: _bucket(r, cfg), axis=1)

    # ---- Capacity check: spill lowest Do-now items to Defer -----------------
    t["capacity_note"] = ""
    do_now = t[t["bucket"] == "Do now"].sort_values("score", ascending=False)
    used = 0.0
    counted_keys: set[str] = set()
    for idx, r in do_now.iterrows():
        key = r["cluster_id"] if (r["classification"] == "reactive" and r["cluster_id"]) else r["request_id"]
        # A cluster's cost is paid once, not per ticket.
        increment = 0.0 if key in counted_keys else float(r["cost_points"])
        if used + increment > cfg["quarter_capacity_points"] and increment > 0:
            t.loc[idx, "bucket"] = "Later"
            t.loc[idx, "capacity_note"] = "over capacity"
            continue
        used += increment
        counted_keys.add(key)

    t["score"] = t["score"].round(2)
    t["impact"] = t["impact"].round(2)
    t["cost_share"] = t["cost_share"].round(2)
    t["bucket"] = pd.Categorical(t["bucket"], categories=BUCKET_ORDER, ordered=True)
    t = t.sort_values(["bucket", "score"], ascending=[True, False]).reset_index(drop=True)

    cl_out = pd.DataFrame(cluster_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return t, cl_out


def _bucket(r, cfg) -> str:
    if r["redirect"]:
        return "Hand off"
    if (
        r["eff_effort_bucket"] == "unclear"
        and r["eff_confidence"] < cfg["discovery_confidence_max"]
        and r["impact"] >= cfg["discovery_min_impact"]
    ):
        return "Investigate first"
    if r["score"] >= cfg["do_now_threshold"]:
        return "Do now"
    if r["score"] >= cfg["do_now_threshold"] * cfg["defer_ratio"]:
        return "Later"
    return "Not this quarter"


def capacity_split(scored: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Reactive vs proactive share of committed points (Do now + Discovery spikes)."""
    committed = scored[scored["bucket"].isin(["Do now", "Investigate first"])]
    by_class = committed.groupby("classification")["cost_share"].sum()
    reactive = float(by_class.get("reactive", 0.0))
    proactive = float(by_class.get("proactive", 0.0))
    total = reactive + proactive
    cap = config["quarter_capacity_points"]
    return {
        "reactive_points": round(reactive, 1),
        "proactive_points": round(proactive, 1),
        "committed_points": round(total, 1),
        "capacity_points": cap,
        "reactive_pct": round(100 * reactive / total) if total else 0,
        "proactive_pct": round(100 * proactive / total) if total else 0,
        "headroom_points": round(cap - total, 1),
        "reactive_in_backlog_pct": round(
            100 * (scored["classification"] == "reactive").mean()
        ),
    }


# --------------------------------------------------------------------------- #
# Diff: what the AI proposed vs what the reviewer finalised
# --------------------------------------------------------------------------- #

DIFF_FIELDS = [
    "classification", "cluster_id", "severity", "confidence",
    "effort_bucket", "redirect", "metric_linked", "retires",
]


def diff_tags(ai_tickets: pd.DataFrame, final_tickets: pd.DataFrame) -> pd.DataFrame:
    a = ai_tickets.set_index("request_id")
    f = final_tickets.set_index("request_id")
    rows = []
    for rid in f.index:
        if rid not in a.index:
            continue
        for field in DIFF_FIELDS:
            av, fv = _norm(a.loc[rid].get(field)), _norm(f.loc[rid].get(field))
            if av != fv:
                rows.append(
                    dict(
                        request_id=rid,
                        field=field,
                        ai_proposed=av,
                        reviewer_final=fv,
                        reviewer_note=f.loc[rid].get("reviewer_note", ""),
                    )
                )
    return pd.DataFrame(rows, columns=["request_id", "field", "ai_proposed", "reviewer_final", "reviewer_note"])


def _norm(v):
    if isinstance(v, list):
        return ", ".join(sorted(v))
    if isinstance(v, float) and pd.isna(v):
        return ""
    if v is None:
        return ""
    if isinstance(v, (bool,)):
        return str(v)
    if isinstance(v, float):
        return round(v, 2)
    return str(v).strip()


def _as_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def merged_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg
