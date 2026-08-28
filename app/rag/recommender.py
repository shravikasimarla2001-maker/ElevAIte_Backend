import json
import re
from app.config import ai_client
from app.rag.db import get_db_connection

def generate_personalized_takeaways(
    target_role: str,
    skill_domain: str,
    user_score: int,
    content_title: str,
    content_snippet: str
) -> list[str]:
    """Uses Gemini 2.5 to synthesize 2-3 'Why Watch/Read This' takeaways mapped to career goals."""
    if not ai_client:
        return [
            f"Directly addresses your current gap in {skill_domain}.",
            f"Essential architectural patterns for aspiring {target_role}s."
        ]

    prompt = f"""
    You are an executive technical mentor. 
    Candidate Target Role: {target_role}
    Evaluated Skill Domain: {skill_domain} (Assessed Current Score: {user_score}/10)
    Recommended Content: "{content_title}"
    Summary Context: "{content_snippet}"
    
    Generate exactly 2 concise, high-impact bullet points explaining "Why You Should Study This".
    Explicitly frame why it bridges their specific skill gap for the {target_role} role.
    
    Return STRICTLY a JSON array of strings:
    ["Takeaway 1...", "Takeaway 2..."]
    """
    try:
        res = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = res.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception:
        return [
            f"Bridges technical gaps in {skill_domain} required for {target_role}.",
            "Covers scalable production implementation nuances."
        ]

def save_recommendations(user_id: str, target_role: str, skill_domain: str, ranked_items: list[dict]):
    """Persists recommendations into PostgreSQL user_recommendations table."""
    with get_db_connection() as conn:
        if not conn:
            return
        with conn.cursor() as cur:
            for item in ranked_items[:3]:
                if not item.get("content_id"):
                    continue
                cur.execute(
                    """
                    INSERT INTO user_recommendations (user_id, content_id, skill_domain, relevance_score, why_watch_takeaways, target_role)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        user_id,
                        item["content_id"],
                        skill_domain,
                        item["relevance_score"],
                        json.dumps(item["takeaways"]),
                        target_role
                    )
                )
            conn.commit()