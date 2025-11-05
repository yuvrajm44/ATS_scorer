# app.py
# Main FastAPI application

from fastapi import FastAPI
from router import router

app = FastAPI(
    title="ATS Resume Scoring API",
    description="Score resumes against job descriptions using AI",
    version="1.0.0"
)

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)