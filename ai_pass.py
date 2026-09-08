"""
AI pass: the AI model reads the whole backlog once and proposes tags.

It does NOT score. It proposes clusters, severity, confidence, classification,
what each proactive item would structurally retire, and flags contradictions.
The reviewer edits those proposals in the app; scoring.py does the arithmetic.

The response is cached to disk so the app runs without an API key and so the
demo output does not drift between runs. (No temperature parameter: SDK 1.0+
rejects it, and the cache is what gives us a stable demo anyway.)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_PATH = Path(__file__).parent / "cache" / "ai_pass.json"
DEFAULT_MODEL = os.environ.get("BET_SCORER_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are helping a Lead Product Manager triage an inbound request backlog for Trellis, a B2B ordering platform connecting food distributors with restaurant and hospitality operators.

Your job is to READ, not to score. You will tag every ticket and propose clusters. A human will review and edit every tag before any scoring happens, and the scoring itself is done by a separate deterministic formula. Do not output numeric priority scores.

Read all records together before tagging any of them. Several tickets may share a root cause that is only visible across records; look for that, but do not force clusters where the evidence is weak.

Definitions you must use:

classification
- "reactive": the item exists because an account reported a problem or asked for something.
- "proactive": self-originated by the product team, usually tied to a company metric.

cluster
- Group reactive tickets that most likely share ONE root cause and ONE fix. A cluster must have at least 2 tickets. Do not cluster by request_type alone; cluster by the mechanism that would be fixed. If two tickets look similar but are probably different bugs, keep them separate and say why in the evidence.
- Each cluster gets one shared effort_bucket (the cost of fixing the root cause once) and one confidence.

severity (per ticket)
- Base this on the MEASURED or DESCRIBED effect in raw_notes, not the tone of the summary or the subject line. "URGENT" in a subject line with "under 2% of orders affected" in the notes is low, not high.
- high: blocks orders or revenue for an account at scale, or carries clear churn / reputational risk, or an exec is involved.
- medium: real operational pain, workarounds exist, or affects a meaningful slice.
- low: cosmetic, rare, a workaround request, or a nice-to-have.
- For proactive items, severity means how urgent the metric problem is that the item addresses.

confidence (0.0 to 1.0)
- How well the root cause and fix are understood. A confirmed cause with a known fix is 0.9 or higher. "Suspect", "possibly", "hard to reproduce", "ambiguous ownership" push it down. For proactive items: how confident you are it would move the named metric.

effort_bucket
- One of: "small", "1-2 sprints", "large", "unclear". Start from rough_effort_hint. If a ticket joins a cluster, the cluster's effort is what matters, not the ticket's.

redirect (boolean)
- true when the item is not product/engineering work at all: a configuration change, a data cleanup, or something support or infra should own. Explain in reason.

metric_linked (boolean)
- true only if the notes name a company metric the item moves.

retires (list of request_ids)
- For proactive items only: which reactive tickets in THIS backlog would this item structurally eliminate or prevent from recurring? Be strict. A self-service setup wizard does not fix a field-mapping bug. Leave empty for reactive items.

ambiguity_note
- Anything the human must decide: contradictions, missing facts, ownership questions, urgency claims that don't match the evidence. Empty string if none.

accounts
- You have no prior knowledge of these accounts. Derive each account's tier ONLY from what the summaries and raw_notes in this backlog say (phrases like "top-5 by GMV", "enterprise tier", "mid-market", "small regional, low GMV"). Return one entry per distinct source_account with a tier of "top", "enterprise", "mid", "small", "internal", or "unknown", and quote the evidence. Use "unknown" when the notes give no tier signal; do not guess from the account name. Treat "Internal" (or any source that is the product team itself) as "internal".

Be specific and terse in every reason. Cite request_ids when you link tickets."""

TOOL_NAME = "submit_backlog_tags"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Submit the proposed tags and clusters for the whole backlog.",
    "input_schema": {
        "type": "object",
        "required": ["accounts", "clusters", "tickets", "cross_record_observations"],
        "properties": {
            "accounts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source_account", "tier", "evidence"],
                    "properties": {
                        "source_account": {"type": "string", "description": "Exactly as it appears in the data."},
                        "tier": {"type": "string", "enum": ["top", "enterprise", "mid", "small", "internal", "unknown"]},
                        "evidence": {"type": "string", "description": "The phrase(s) and request_ids the tier was read from."},
                    },
                },
            },
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cluster_id", "label", "ticket_ids", "root_cause_hypothesis", "evidence", "effort_bucket", "confidence"],
                    "properties": {
                        "cluster_id": {"type": "string", "description": "snake_case id, e.g. lead_time_logic"},
                        "label": {"type": "string"},
                        "ticket_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                        "root_cause_hypothesis": {"type": "string"},
                        "evidence": {"type": "string", "description": "Which phrases in which tickets link them."},
                        "effort_bucket": {"type": "string", "enum": ["small", "1-2 sprints", "large", "unclear"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "tickets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["request_id", "classification", "cluster_id", "severity", "confidence", "effort_bucket", "redirect", "metric_linked", "retires", "reason", "ambiguity_note"],
                    "properties": {
                        "request_id": {"type": "string"},
                        "classification": {"type": "string", "enum": ["reactive", "proactive"]},
                        "cluster_id": {"type": "string", "description": "Empty string if not clustered."},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "effort_bucket": {"type": "string", "enum": ["small", "1-2 sprints", "large", "unclear"]},
                        "redirect": {"type": "boolean"},
                        "metric_linked": {"type": "boolean"},
                        "retires": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string", "description": "One line. Why these tags."},
                        "ambiguity_note": {"type": "string"},
                    },
                },
            },
            "cross_record_observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns only visible by reading across records.",
            },
        },
    },
}


def backlog_as_text(df: pd.DataFrame) -> str:
    cols = ["request_id", "date_received", "source_account", "request_type", "summary", "raw_notes", "rough_effort_hint"]
    lines = ["Backlog (27 records). One record per block.", ""]
    for _, r in df[cols].iterrows():
        lines.append(f"### {r['request_id']}")
        for c in cols[1:]:
            lines.append(f"{c}: {r[c]}")
        lines.append("")
    return "\n".join(lines)


def prompt_hash(df: pd.DataFrame, model: str, system_prompt: str | None = None) -> str:
    h = hashlib.sha256()
    h.update((system_prompt or SYSTEM_PROMPT).encode())
    h.update(json.dumps(TOOL_SCHEMA, sort_keys=True).encode())
    h.update(backlog_as_text(df).encode())
    h.update(model.encode())
    return h.hexdigest()[:12]


def run_ai_pass(df: pd.DataFrame, api_key: str, model: str = DEFAULT_MODEL, system_prompt: str | None = None) -> dict:
    """Call the model with a forced tool so the output is schema-valid JSON."""
    import anthropic  # imported here so the app runs without the SDK installed

    system_prompt = system_prompt or SYSTEM_PROMPT
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        backlog_as_text(df)
        + "\n\nTag every one of the 27 tickets and propose clusters. "
        "Use the submit_backlog_tags tool for your entire answer."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system_prompt,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": user_msg}],
    )
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    payload = tool_block.input
    payload = validate_payload(payload, df)
    return {
        "source": "api",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_hash": prompt_hash(df, model, system_prompt),
        "system_prompt": system_prompt,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        "result": payload,
    }


def validate_payload(payload: dict, df: pd.DataFrame) -> dict:
    """Light checks: every ticket present once, cluster ids consistent."""
    ids = list(df["request_id"])
    got = [t["request_id"] for t in payload["tickets"]]
    missing = [i for i in ids if i not in got]
    extra = [i for i in got if i not in ids]
    if missing or extra:
        raise ValueError(f"AI pass ticket mismatch. missing={missing} extra={extra}")
    # Every account in the data gets a tier entry; unknown if the model skipped it.
    seen = {a["source_account"] for a in payload.get("accounts", [])}
    for acct in sorted(set(df["source_account"])):
        if acct not in seen:
            payload.setdefault("accounts", []).append({"source_account": acct, "tier": "unknown", "evidence": "not returned by model"})
    cluster_ids = {c["cluster_id"] for c in payload["clusters"]}
    for t in payload["tickets"]:
        if t.get("cluster_id") and t["cluster_id"] not in cluster_ids:
            t["ambiguity_note"] = (t.get("ambiguity_note", "") + f" [cluster_id '{t['cluster_id']}' not in cluster list; cleared]").strip()
            t["cluster_id"] = ""
    return payload


def cache_path_for(hash_: str) -> Path:
    return CACHE_PATH.parent / f"ai_pass_{hash_}.json"


def save_cache(blob: dict, path: Path | None = None) -> None:
    """Save under the prompt hash so different uploads never overwrite each other."""
    path = path or cache_path_for(blob["prompt_hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def load_cache(df: pd.DataFrame | None = None, model: str = DEFAULT_MODEL, system_prompt: str | None = None) -> dict | None:
    """
    Return the cached proposal for THIS dataset: the hash-named file if it
    exists, otherwise the bundled default only if it covers the same tickets.
    """
    if df is not None:
        hashed = cache_path_for(prompt_hash(df, model, system_prompt))
        if hashed.exists():
            return json.loads(hashed.read_text(encoding="utf-8"))
    if not CACHE_PATH.exists():
        return None
    blob = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if df is None:
        return blob
    cached_ids = {t["request_id"] for t in blob["result"]["tickets"]}
    return blob if cached_ids == set(df["request_id"]) else None


def payload_to_frames(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickets = pd.DataFrame(payload["tickets"])
    clusters = pd.DataFrame(payload["clusters"])
    if len(clusters):
        clusters["ticket_ids"] = clusters["ticket_ids"].apply(lambda x: ", ".join(x))
    return tickets, clusters


if __name__ == "__main__":
    # CLI: python ai_pass.py data/Inbound_Requests.csv
    import sys
    from scoring import read_backlog

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY to run the AI pass from the CLI.")
    src = sys.argv[1] if len(sys.argv) > 1 else "data/Inbound_Requests.csv"
    blob = run_ai_pass(read_backlog(src), key)
    save_cache(blob)
    save_cache(blob, CACHE_PATH)  # also refresh the bundled default
    print(f"Saved {CACHE_PATH} and {cache_path_for(blob['prompt_hash'])} ({blob['usage']})")
