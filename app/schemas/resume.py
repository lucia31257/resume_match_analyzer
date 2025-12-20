from pydantic import BaseModel


class ReviewRequest(BaseModel):
    resume_text: str
    jd_text: str


class ReviewResponse(BaseModel):
    score: int
    reason: str
