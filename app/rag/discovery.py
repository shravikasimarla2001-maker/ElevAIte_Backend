import os
import json
import httpx
from app.config import ai_client

CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY", os.getenv("GEMINI_API_KEY", ""))
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")  # Programmable Search Engine CX ID

async def generate_search_queries_for_gaps(skill_domain: str, score: int, target_role: str) -> list[str]:
    """Uses Gemini to formulate highly specific search queries for user's weakest skill domains."""
    if not ai_client:
        return [f"{target_role} {skill_domain} production best practices architecture"]

    prompt = f"""
    You are an AI learning curator. A candidate targeting the role "{target_role}" has an assessed score of {score}/10 in "{skill_domain}".
    Generate exactly 2 targeted, highly technical search queries to discover high-value tutorials, official documentation, or technical videos that bridge this specific gap.
    
    Return STRICTLY a JSON array of strings:
    ["query 1", "query 2"]
    """
    try:
        res = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = res.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception:
        return [f"Google Cloud {skill_domain} deep dive architecture guide", f"{target_role} {skill_domain} hands-on tutorial"]

async def search_google_custom(query: str) -> list[dict]:
    """Queries Google Custom Search API or YouTube search endpoints for relevant content."""
    if not SEARCH_ENGINE_ID or not CUSTOM_SEARCH_API_KEY:
        # Structured fallback mock results if Search Engine ID is pending
        return [
            {
                "title": f"Production Architecture Patterns for {query.split()[0]}",
                "url": f"[https://cloud.google.com/blog/topics/developers-practitioners/](https://cloud.google.com/blog/topics/developers-practitioners/){query.replace(' ', '-').lower()}",
                "snippet": f"A comprehensive technical deep-dive into scalable, resilient architectures for {query}.",
                "source_type": "blog"
            },
            {
                "title": f"Deep Dive: Building Enterprise Systems with {query.split()[0]}",
                "url": f"[https://www.youtube.com/watch?v=mock](https://www.youtube.com/watch?v=mock)_{query.replace(' ', '_')[:10]}",
                "snippet": f"Hands-on architectural guide and code walk-through for {query}.",
                "source_type": "youtube"
            }
        ]

    url = "[https://www.googleapis.com/customsearch/v1](https://www.googleapis.com/customsearch/v1)"
    params = {
        "key": CUSTOM_SEARCH_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 3
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        if resp.status_code != 200:
            return []
        data = resp.json()
        
        items = []
        for item in data.get("items", []):
            items.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source_type": "youtube" if "youtube.com" in item.get("link", "") else "blog"
            })
        return items