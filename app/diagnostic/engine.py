import json
import re
from app.config import ai_client

# Explicit model ID as required by Google GenAI
MODEL_ID = "gemini-3.6-flash"


def clean_json_response(raw_text: str) -> dict:
    """Strips markdown formatting and parses raw JSON cleanly."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text.strip())

def generate_first_question(target_role: str, resume_context: str, skill_matrix: dict) -> dict:
    """Generates real-time Question #1 using Gemini 3.6 Flash."""
    if not ai_client:
        raise ValueError("GEMINI_API_KEY environment variable is missing or invalid.")

    prompt = f"""
    You are an expert technical interviewer assessing a candidate for: {target_role}.
    Current Skill Matrix Status: {json.dumps(skill_matrix)}
    Candidate Background Context: {resume_context[:2000]}
    
    Generate Question #1: A foundational architecture or design question to establish baseline capability.
    Return ONLY a JSON object with keys: "question", "skill_domain", "difficulty".
    """
    
    response = ai_client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return clean_json_response(response.text)


def evaluate_and_generate_next(
    target_role: str,
    question_number: int,
    previous_question: str,
    user_answer: str,
    skill_domain: str,
    current_skill_matrix: dict
) -> dict:
    """Evaluates answer, scores it (1-10), and generates Question N+1."""
    if not ai_client:
        raise ValueError("GEMINI_API_KEY environment variable is missing or invalid.")

    prompt = f"""
    You are an expert technical interviewer assessing a candidate for: {target_role}.
    Current Skill Matrix: {json.dumps(current_skill_matrix)}
    
    Last Question Asked ({skill_domain}): "{previous_question}"
    Candidate's Answer: "{user_answer}"
    
    Tasks:
    1. Score the answer for '{skill_domain}' from 1 to 10.
    2. Provide 1 concise sentence of direct constructive feedback.
    3. Generate Question #{question_number + 1} targeting the next skill domain.
    
    Return ONLY a JSON object with keys: "score", "feedback", "evaluated_domain", "next_question", "next_skill_domain".
    """

    response = ai_client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return clean_json_response(response.text)