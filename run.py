import uvicorn
from app.main import app  # noqa: F401 — re-exported for `uvicorn run:app`

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
