# Bet Scorer

Scores an inbound request backlog and recommends a reactive / proactive capacity split for next quarter.

The design principle: the AI model reads, a human decides, code does the arithmetic.

1. Ingest: reads the backlog CSV (handles the cp1252 encoding of the companion file).
2. AI pass: one API call with the whole backlog. The model proposes root-cause clusters, severity, confidence, classification, what each proactive item would retire, and flags contradictions. It is instructed not to score. Output is schema-enforced via a forced tool call and cached to `cache/ai_pass.json`.
3. Review: the prompt can be edited before the AI pass runs. Every proposed tag is editable in a table. Scoring weights are sliders in the sidebar. Results update live.
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

or paste the key into the sidebar and click "Run AI pass via API". The model defaults to `claude-sonnet-4-6`; override with `BET_SCORER_MODEL`.

Note: the `cache/ai_pass.json` in this repo is a seed drafted offline from the same prompt definitions (`source: seed`), not a live model run. The app warns when it is loaded. Re-run the AI pass before submitting so the cache is a real model output.

## Deploy a public link (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo. `.gitignore` already excludes `.streamlit/secrets.toml`, so the key never lands in git.
2. Go to https://share.streamlit.io, sign in with GitHub, click "Create app", pick the repo, branch `main`, main file `app.py`.
3. Under Advanced settings > Secrets, paste: `ANTHROPIC_API_KEY = "sk-ant-..."`. Deploy.
4. Open the app, click "Run AI pass via API" on the bundled data once, download nothing yet, then commit the refreshed `cache/ai_pass.json` (run `python ai_pass.py` locally to write it) so visitors see a real model output before they run anything.

When a key is present in Secrets the sidebar hides the key field and shows "API key is configured on the server". Each browser session can trigger the AI pass at most 3 times (`MAX_RUNS_PER_SESSION` in `app.py`). Set a monthly spend limit on the key in the API console before sharing the link publicly. Cached proposals are saved per dataset (`cache/ai_pass_<hash>.json`) and the host's disk is ephemeral, so they reset on redeploy; that is fine, it is a cache.

## Scoring

Reactive tickets are scored at the root-cause cluster level, not one by one:

    cluster score = sum(account_weight x severity over member tickets) x confidence / effort_points

Each member inherits the cluster score and carries an equal share of the cluster's cost. A singleton is a cluster of one.

Proactive items:

    score = (own impact + impact of the reactive tickets it retires) x metric_bonus x confidence / effort_points

Buckets, checked in order: Redirect (not product work) -> Discovery spike (effort unclear, low confidence, enough impact to be worth a spike) -> Do now (score >= threshold) -> Defer (within the defer band) -> Decline. Do-now items spill to Defer when committed points exceed quarter capacity.

The split is the share of committed points (Do now + Discovery spike) by classification. Redirected work is excluded because it is not roadmap capacity.

Every number in the formula lives in `DEFAULT_CONFIG` in `scoring.py` and is exposed in the sidebar.

## Files

- `app.py`: Streamlit app.
- `ai_pass.py`: prompt, JSON schema, API call, cache. Runs standalone as a CLI.
- `scoring.py`: deterministic scoring, bucketing, capacity split, AI-vs-reviewer diff.
- `report.py`: builds `summary.md`.
- `data/Inbound_Requests.csv`: the backlog.
- `cache/ai_pass.json`: the AI proposal the app loads.

## Assumptions to state in the rationale

- Account weights are inferred from tier language in the notes; GMV figures are not in the data.
- Quarter capacity (30 points) assumes a five-person team, six two-week sprints, roughly five points per sprint after support load.
- Effort hints map to points 1 / 3 / 8; "unclear" becomes a one-point discovery spike rather than a guess.
- Clusters are hypotheses about shared root causes. They are the highest-leverage judgement in the tool and the first thing support would need to verify.
