# AutoEDA Backend

FastAPI Python backend for Exploratory Data Analysis API.

## Quick Start

### 1. Clone and Install
```bash
git clone <autoeda-backend-repo>
cd autoeda-backend
pip install -r requirements.txt
```

On Mac, use `pip3`:
```bash
pip3 install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file in the root directory:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/autoeda
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin-password
GEMINI_API_KEY=your-gemini-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
```

### 3. Run Migrations
```bash
python -m alembic upgrade head
```

Or on Mac:
```bash
python3 -m alembic upgrade head
```

### 4. Start Development Server
```bash
python run.py
```

Or on Mac:
```bash
python3 run.py
```

Server runs on [http://localhost:8000](http://localhost:8000)

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

## Environment Variables

- `SECRET_KEY` - JWT secret key (required)
- `DATABASE_URL` - PostgreSQL connection string (required)
- `ADMIN_EMAIL` - Admin user email
- `ADMIN_PASSWORD` - Admin user password
- `GEMINI_API_KEY` - Google Gemini API key
- `ALGORITHM` - JWT algorithm (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES` - Token expiry (default: 480)

## Tech Stack

- Python 3.11+
- FastAPI
- SQLAlchemy
- PostgreSQL
- DuckDB
- pandas, numpy, scipy, scikit-learn
