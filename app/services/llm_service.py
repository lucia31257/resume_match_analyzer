from app.core.config import settings
from openai import OpenAI
import json

client = OpenAI(
    api_key=settings.API_KEY,
    base_url="https://api.deepseek.com"
)


def analyze_resume(resume: str, jd: str):
    prompt = f"""
    You are a resume screening assistant.

    Given a resume and a job description, provide:
    1. A matching score between 0 and 100 (higher means better match)
    2. A short reason explaining the score

    Output strictly in JSON format like:
    {{"score": 85, "reason": "The candidate has strong experience in X, Y, Z."}}

    Resume:
    {resume}

    Job Description:
    {jd}
    """
    system_prompt = """
    You are an HR assistant specialized in resume screening.
    Your task is to evaluate a resume against a job description.
    you should:
    1. Score the match between 0 and 100 (higher is better)
    2. Give a short reason explaining the score
    
    Output Rules:
    1. Output strictly in JSON format.
    2. Do NOT wrap the JSON in markdown code blocks (like ```json).
    3. The JSON keys must be: "score" (0-100 integer) and "reason" (string).
    
    An example of output:
    {"score": 85, "reason": "Candidate has strong experience in X, Y, Z."}
    """

    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        result = json.loads(content)

    except json.JSONDecodeError:
        result = {"score": None, "reason": content}

    return result
