"""Main entry point for the backend service."""

from fastapi import FastAPI

app = FastAPI(title="Advanced Medical AI")


@app.get("/")
def read_root():
    return {"message": "Advanced Medical AI backend is running"}
