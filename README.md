# AutoEDA Backend

FastAPI backend for automated exploratory data analysis: dataset ingestion, statistical analysis, a tool-calling AI agent (Scout), and evidence-backed hypothesis testing.

## Quick Start

### 1. Install

```bash
pip3 install -r requirements.txt
```

### 2. Configure Environment

Create `.env` in the project root:

```bash
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost:5432/autoeda
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin-password
ANTHROPIC_API_KEY=your-claude-key
```

### 3. Run Migrations

```bash
python3 -m alembic upgrade head
```

### 4. Start

```bash
python3 run.py
```

Server: [http://localhost:8000](http://localhost:8000) · API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Production runs via the included `Dockerfile` (`alembic upgrade head` then `uvicorn app.main:app`).

## Environment Variables

- `SECRET_KEY` — JWT signing key; also derives the key used to encrypt stored data source credentials
- `DATABASE_URL` — PostgreSQL connection string
- `ADMIN_EMAIL` / `ADMIN_PASSWORD` — seeded admin account
- `ALGORITHM` (default `HS256`), `ACCESS_TOKEN_EXPIRE_MINUTES` (default `480`)
- `AUTO_PROVISION_EMAIL_DOMAIN` — emails on this domain get an account auto-created on first login
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` — at least one; the active provider is picked by `app/ai/llm.py` in that priority order
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` / `S3_ATTACHMENTS_BUCKET` — large uploads (attachments, Scout images) go browser/client → S3 directly via presigned URLs, bypassing the frontend proxy's body-size limit
- `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET`, `SHAREPOINT_EXCEL_URL` — existing SharePoint integration (`app/integrations/sharepoint.py`)

## What's in here

**EDA** (`app/eda/`): profiling, correlations (Pearson/Spearman/Kendall, Cramér's V, η², significance-gated insights), missing-value analysis, outliers, feature importance (RF/MI/ANOVA/permutation/SHAP + redundancy/leakage detection + minimal-feature-set finder), distributions, time series, text analysis, statistical tests.

**Scout** (`app/ai/agent/`): a tool-calling agent — profiling, correlations, SQL (single-dataset and workspace-wide), sandboxed Python, real statistical tests — streamed over SSE with visible tool-by-tool progress. Provider-agnostic (Claude/OpenAI/Gemini) via `app/ai/llm.py`.

**Hypotheses** (`app/ai/agent/hypothesis_orchestrator.py`): reuses Scout's tool-calling loop with a read-only tool allowlist to validate a claim, or generate pre-verified ones, against an actual computed test instead of narration.

**Data Sources** (`app/connectors/`): pluggable connectors (databases, cloud storage, REST APIs) behind a single registry; credentials are encrypted at rest.

**Heavy computation isolation** (`app/process_pool.py`): CPU/memory-heavy analysis runs in a separate process pool so one crash or OOM can't take down the API server.

## Project Structure

```
app/
  routers/        One file per resource (datasets, scout, hypotheses, sources, warehouse, sql_editor, ...)
  models/         SQLAlchemy models
  schemas/        Pydantic request/response schemas
  eda/            Statistical analysis implementations
  ai/
    providers/    Claude / OpenAI / Gemini, behind a shared interface
    agent/        Scout's and Hypotheses' tool-calling orchestration + tool implementations
  connectors/     Data source connectors + registry
  integrations/   Standalone third-party integrations (SharePoint)
  core/           Event bus, presence
alembic/          DB migrations
```

## Tech Stack

- Python 3.11+, FastAPI, SQLAlchemy, Alembic, PostgreSQL
- pandas, numpy, scipy, scikit-learn, shap, statsmodels, ruptures
- boto3, azure-storage-blob, google-cloud-storage/bigquery, snowflake-connector-python, databricks-sql-connector
- anthropic, openai, google-genai
