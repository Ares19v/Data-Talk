"""
app.py — Root entry point.
For production: serve the built React dist from FastAPI.
For dev: run FastAPI on :8000, Vite on :5173.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from api.app import app  # noqa: F401 — expose for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
