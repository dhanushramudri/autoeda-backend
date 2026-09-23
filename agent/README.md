# AutoEDA Local Training Agent

Trains the models you queue on the AutoML page — but on **your own laptop**,
not the cloud server. No cloud compute bill for training, no Python
environment to set up by hand — just Docker.

## Run it

```bash
docker build -t autoeda-agent .
docker run --rm \
  -e AUTOEDA_API_URL="http://<your-autoeda-host>:8000/api/v1" \
  -e AUTOEDA_EMAIL="you@jmangroup.com" \
  autoeda-agent
```

No password — AutoEDA accounts are email-only (the same passwordless login the web app itself uses).

Leave it running in a terminal. It polls for any experiment you've queued
from the AutoML page in the app, trains it, and reports results back —
you'll see the leaderboard fill in live in the browser. Ctrl+C to stop; it
picks up any not-yet-claimed experiments again next time you start it.

## What it actually does

For each queued experiment it:
1. Downloads the dataset via the same export endpoint the app itself uses.
2. Shortlists 2–3 algorithms using the DS Sales Playbook's own heuristic
   (dataset size × how categorical the features are) — see `algorithms.py`.
3. Splits 70:30, trains each shortlisted model, evaluates it.
4. Reports each run's metrics back immediately (so you see progress as it
   happens, not just at the end), then uploads the trained model file.

## Models it can train

Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting, SVM,
CatBoost — whichever the playbook's heuristic shortlists for your dataset's
size and shape. Classification or regression targets both work; the
algorithm keys map to their standard regression counterpart automatically.
