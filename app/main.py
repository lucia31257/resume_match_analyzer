from fastapi import FastAPI
from app.services.llm_service import analyze_resume
from app.schemas.resume import ReviewRequest, ReviewResponse
from app.services.vector_service import calculate_similarity
app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/score", response_model=ReviewResponse)
def get_score(request: ReviewRequest):
    result = analyze_resume(request.resume_text, request.jd_text)
    return result


@app.post("/match", response_model=dict)
def match_resume(request: ReviewRequest):
    result = calculate_similarity(request.resume_text, request.jd_text)
    return {
        "score": result,
        "type": "vector_similarity",
        "reason": "Based on semantic distance (Fast Check)"
    }
