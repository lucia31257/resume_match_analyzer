from fastapi import FastAPI
from app.services.llm_service import analyze_resume
from app.schemas.resume import ReviewRequest, ReviewResponse
app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/score", response_model=ReviewResponse)
def get_score(request: ReviewRequest):
    result = analyze_resume(request.resume_text, request.jd_text)
    return result
