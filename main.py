# app.py
# Main FastAPI application

from fastapi import FastAPI
from router import router
import os

app = FastAPI(
    title="ATS Resume Scoring API",
    description="Score resumes against job descriptions using AI",
    version="1.0.0"
)

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)