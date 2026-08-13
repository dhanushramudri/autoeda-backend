# AutoEDA Backend

FastAPI backend for automated exploratory data analysis: dataset ingestion, statistical analysis, a tool-calling AI agent (Scout), and evidence-backed hypothesis testing.

## Prerequisites

- Python 3.11+
- PostgreSQL (recommended for production) **or** SQLite (zero-setup, local dev only)
- Redis (optional — used for result caching; app works without it)

## Quick Start (Local)

### 1. Create a virtual environment

```bash
cd autoeda-backend

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy the example and edit:

```bash
cp .env.example .env
```

Minimum `.env` for local development (SQLite — no database setup needed):

```env
SECRET_KEY=any-random-string-at-least-32-chars

# SQLite — easiest for local dev
DATABASE_URL=sqlite:///./app/storage/autoeda.db

ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=Admin@123

# At least one AI key is needed for Scout and Hypotheses features
GEMINI_API_KEY=your-gemini-key
```

> For PostgreSQL: `DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/autoeda`

### 4. Run migrations

```bash
alembic upgrade head
```

> On a fresh install, `alembic upgrade head` and the auto-create on startup both work. Use Alembic going forward for any schema changes.

### 5. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

The admin account from `.env` is seeded automatically on first start.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | Yes | JWT signing key (any random string, min 32 chars) |
| `DATABASE_URL` | Yes | SQLite (`sqlite:///./app/storage/autoeda.db`) or PostgreSQL connection string |
| `ADMIN_EMAIL` | Yes | Email for the auto-seeded admin account |
| `ADMIN_PASSWORD` | Yes | Password for the admin account |
| `GEMINI_API_KEY` | No* | Google Gemini — Scout AI & Hypotheses |
| `OPENAI_API_KEY` | No* | OpenAI alternative for AI features |
| `ANTHROPIC_API_KEY` | No* | Claude alternative for AI features |
| `ADMIN_EMAILS` | No | Comma-separated additional admin emails |
| `MICROSOFT_EMAILS` | No | Comma-separated emails for Microsoft mock login |
| `AUTO_PROVISION_EMAIL_DOMAIN` | No | Emails on this domain get an account auto-created on first login (default: `jmangroup.com`) |
| `GLOBAL_DATASET_EMAIL` | No | Datasets uploaded by this account are shared across all workspaces |
| `AWS_ACCESS_KEY_ID` | No | S3 large file uploads (attachments) |
| `AWS_SECRET_ACCESS_KEY` | No | S3 |
| `AWS_REGION` | No | S3 region (default: `eu-north-1`) |
| `S3_ATTACHMENTS_BUCKET` | No | S3 bucket name |
| `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` | No | Azure AD / SharePoint integration |
| `SHAREPOINT_EXCEL_URL` | No | SharePoint feedback table URL |
| `EDA_POOL_TIMEOUT_SECONDS` | No | Timeout for heavy EDA computations (default: `600`) |

> *At least one of `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` is required for Scout and Hypotheses. The app falls back through them in that priority order.

---

## What's in here

**EDA**: profiling, correlations (Pearson/Spearman/Kendall, Cramér's V, η², significance-gated insights), missing-value analysis, outliers, feature importance (RF/MI/ANOVA/permutation/SHAP + redundancy/leakage detection + minimal-feature-set finder), distributions, time series, text analysis, statistical tests.

**Scout**: a tool-calling agent — profiling, correlations, SQL (single-dataset and workspace-wide), sandboxed Python, real statistical tests — streamed over SSE with visible tool-by-tool progress. Provider-agnostic (Claude/OpenAI/Gemini).

**Hypotheses**: reuses Scout's tool-calling loop with a read-only tool allowlist to validate a claim, or generate pre-verified ones, against an actual computed test instead of narration.

**Data Sources**: pluggable connectors (databases, cloud storage, REST APIs) behind a single registry; credentials are encrypted at rest.

**Heavy computation isolation**: CPU/memory-heavy analysis runs in a separate process pool so one crash or OOM can't take down the API server.

---

## Architecture

**External services:**
- **PostgreSQL** — primary database (SQLAlchemy + Alembic migrations).
- **AWS S3** — presigned-URL uploads/downloads for large files, bypassing the frontend proxy's body-size limit.
- **Claude / OpenAI / Gemini** — LLM providers behind one interface; whichever key is set first (in that priority order) is the active provider.
- **Azure AD / SharePoint** — service-principal auth for the SharePoint integration.

**In-process (no external services needed):**
- **DuckDB** — SQL engine behind Warehouse and SQL Editor; runs in-process against loaded DataFrames.
- **A bounded process pool** — isolates CPU/memory-heavy EDA computation from the main API process.
- **A thread pool + in-memory event bus** — background jobs and real-time notifications.

**Deployment**: Docker container on EC2, built from the included `Dockerfile`.

---

## Project Structure

```
app/
  routers/        One file per resource (datasets, scout, hypotheses, sources, warehouse, sql_editor, ...)
  models/         SQLAlchemy ORM models
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

## Supported Data Sources

- **Files**: CSV, Excel (.xlsx/.xls), JSON, Parquet, TSV
- **Databases**: PostgreSQL, MySQL, SQLite, MSSQL, MongoDB
- **Cloud**: AWS S3, Azure Blob Storage, Google Cloud Storage
- **API**: REST endpoints

---

## Tech Stack

- Python 3.11+, FastAPI, SQLAlchemy, Alembic, PostgreSQL
- pandas, numpy, scipy, scikit-learn, shap, statsmodels, ruptures
- boto3, azure-storage-blob, google-cloud-storage
- anthropic, openai, google-genai

