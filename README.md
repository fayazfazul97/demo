# Bet Scorer

Scores an inbound request backlog and recommends a reactive / proactive capacity split for next quarter.

The design principle: the AI model reads, a human decides, code does the arithmetic.

1. Ingest: reads the backlog CSV (handles the cp1252 encoding of the companion file).
2. AI pass: one API call with the whole backlog. The model proposes root-cause clusters, severity, confidence, classification, what each proactive item would retire, and flags contradictions. It is instructed not to score. Output is schema-enforced via a forced tool call and cached to `cache/ai_pass.json`.
3. Review: the prompt can be edited before the AI pass runs. The model's account sizes, clusters and ticket tags are all editable in tables. Other settings are in the sidebar. Results update live.
4. Finalise: locks the tags, produces the scored CSV, a markdown summary, a diff log of every field the reviewer changed from the AI proposal, and the finalised tags + config as JSON.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

Then click "Load cached proposal". No API key is needed to run on the cached proposal.

To regenerate the AI proposal from the model:

```
export ANTHROPIC_API_KEY=sk-ant-...
python ai_pass.py data/Inbound_Requests.csv      # CLI, writes cache/ai_pass.json
```

or paste the key into the sidebar and click "Run AI pass via API". The model defaults to `claude-sonnet-4-6`; override with `BET_SCORER_MODEL`. The output token limit defaults to 20000; override with `BET_SCORER_MAX_TOKENS` if a large backlog gets cut off.

Note: the `cache/ai_pass.json` in this repo is a seed drafted offline from the same prompt definitions (`source: seed`), not a live model run. The app warns when it is loaded. Re-run the AI pass before submitting so the cache is a real model output.

## Deploy a public link (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo. `.gitignore` already excludes `.streamlit/secrets.toml`, so the key never lands in git.
2. Go to https://share.streamlit.io, sign in with GitHub, click "Create app", pick the repo, branch `main`, main file `app.py`.
3. Under Advanced settings > Secrets, paste: `ANTHROPIC_API_KEY = "sk-ant-..."`. Deploy.
4. Open the app, click "Run AI pass via API" on the bundled data once, download nothing yet, then commit the refreshed `cache/ai_pass.json` (run `python ai_pass.py` locally to write it) so visitors see a real model output before they run anything.

When a key is present in Secrets the sidebar hides the key field and shows "API key is configured on the server". Each browser session can trigger the AI pass at most 3 times (`MAX_RUNS_PER_SESSION` in `app.py`). Set a monthly spend limit on the key in the API console before sharing the link publicly. Cached proposals are saved per dataset (`cache/ai_pass_<hash>.json`) and the host's disk is ephemeral, so they reset on redeploy; that is fine, it is a cache.

## Reducing run-to-run variance

- Temperature is sent as 0 (via `extra_body`, since the 1.0 SDK removed the keyword). Override with `BET_SCORER_TEMPERATURE`.
- Confidence is anchored to four named levels (0.9 / 0.7 / 0.5 / 0.3) with a rubric, so it is a lookup rather than a guess.
- "Consensus of 3 runs" in step 2 runs the analysis three times and keeps what a majority agrees on: majority on categorical fields, median on confidence, clusters only where two runs put the tickets together, elimination links only where two runs propose them. Disagreements are written into the ticket's flag. Three times the cost.

## The author's baseline

`cache/baseline.json` is the fixed analysis the written rationale refers to. When a reviewer loads the bundled backlog it is loaded automatically and labelled as the author's baseline. Reviewers can run their own analysis (the model's reading varies between runs), but their run is kept separately and never replaces the baseline.

To create or update the baseline: run an analysis, optionally review and finalise, then download "Baseline bundle" from step 5 (or "Analysis JSON" from step 2 for the raw analysis without review edits) and commit it as `cache/baseline.json`.

## Scoring

Reactive tickets are scored at the root-cause cluster level, not one by one:

    cluster score = sum(account_weight x severity over member tickets) x confidence / effort_points

Each member inherits the cluster score and carries an equal share of the cluster's cost. A singleton is a cluster of one.

Proactive items:

    score = (own impact + impact of the reactive tickets it retires) x metric_bonus x confidence / effort_points

metric_bonus defaults to 1.0 (off): urgency is already in severity. Handed-off cluster members add no impact to the cluster and carry none of its cost.

Categories, checked in order: Hand off (not engineering work) -> Investigate first (effort unclear and impact above the floor; unclear effort is never scored as known) -> Do now (score >= priority threshold) -> Later (within the band below the threshold) -> Not this quarter. Do-now items spill to Later when committed points exceed quarter capacity.

The split is the share of allocated effort points (Do now + Investigate first) by classification. Spare capacity is filled from Later (configurable) so the quarter is fully allocated. Handed-off work is excluded because it is not roadmap capacity.

Every number in the formula lives in `DEFAULT_CONFIG` in `scoring.py` and is exposed in the sidebar.

## Files

- `app.py`: Streamlit app.
- `ai_pass.py`: prompt, JSON schema, API call, cache. Runs standalone as a CLI.
- `scoring.py`: deterministic scoring, bucketing, capacity split, AI-vs-reviewer diff.
- `report.py`: builds `summary.md`.
- `data/Inbound_Requests.csv`: the backlog.
- `cache/ai_pass.json`: the AI proposal the app loads.

## Assumptions to state in the rationale

- Account weights (1 to 5) are suggested by the model from size clues in the notes and edited by the reviewer in step 3a; GMV figures are not in the data.
- Quarter capacity (30 effort points, adjustable in the sidebar) assumes a five-person team, six two-week sprints, roughly five points per sprint after support load.
- Effort hints map to effort points 1 / 3 / 8; "unclear" becomes a one-point investigation rather than a guess. Severity (1 / 2 / 3) is a multiplier on impact, not a unit of work.
- Clusters are hypotheses about shared root causes. They are the highest-leverage judgement in the tool and the first thing support would need to verify.
