import json
import logging
import re
from typing import List, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from app.config import db, ai_client
from app.rag.discovery import search_google_custom
from google.genai import types
import asyncio
from urllib.parse import urlparse, urlunparse

from app.services.youtube_service import YouTubeDiscoveryService
from app.services.gemini_search_service import GeminiGroundedSearchService
from app.services.enrichment_service import RecommendationEnrichmentService

# Initialize clients (Fail immediately on startup if misconfigured)
youtube_service = YouTubeDiscoveryService()
gemini_search_service = GeminiGroundedSearchService()
enrichment_service = RecommendationEnrichmentService()

logger = logging.getLogger("rag.routes")
router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations & Horizon"])


def clean_url(raw_url: str) -> str:
    """Strips tracking query parameters (utm_*, etc.) from discovery URLs."""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

import asyncio
import logging

logger = logging.getLogger(__name__)

async def run_discovery_pipeline(target_role: str, skill_matrix: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    Orchestrator pipeline configured for a skill_matrix structured as a list of tuples: 
    [('Domain Name', score), ...]
    """
    if not skill_matrix:
        return {
            "total_found": 0,
            "articles_found": 0,
            "videos_found": 0,
            "domains_requested": 0,
            "domains_covered_count": 0,
            "uncovered_domains": [],
            "items": []
        }

    # Run both operations concurrently, passing the list of tuples directly
    doc_items, video_items = await asyncio.gather(
        gemini_search_service.search_articles_batch(
            target_role=target_role, 
            domain_skills=skill_matrix
        ),
        asyncio.to_thread(
            youtube_service.search_videos_batch,
            target_role=target_role,
            domain_skills=skill_matrix,
            max_results=4
        )
    )

    raw_items = []
    raw_items.extend(doc_items)
    raw_items.extend(video_items)

    logger.info(f"raw_items: {raw_items}")

    # Extract domain names correctly from the list of tuples
    domain_names = {domain for domain, _ in skill_matrix}
    covered_domains = {
        domain 
        for item in raw_items 
        for domain in item.get("skill_domains", []) 
        if domain in domain_names
    }

    return {
        "total_found": len(raw_items),
        "articles_found": len(doc_items),
        "videos_found": len(video_items),
        "domains_requested": len(domain_names),
        "domains_covered_count": len(covered_domains),
        "uncovered_domains": list(domain_names - covered_domains),
        "items": raw_items
    }
@router.api_route("/radar/discover/{user_id}", methods=["GET", "POST"], status_code=status.HTTP_200_OK)
@router.post("/radar/discover/v2/{user_id}", status_code=status.HTTP_200_OK)
async def scan_and_recalibrate_radar(user_id: str) -> Dict[str, Any]:
    """
    Unified endpoint for Discovery and Recalibration:
    1. Reads candidate profile (skills, domain scores, target role/level).
    2. Uses Gemini 2.5 Flash with Google Search Grounding to find real, active opportunities.
    3. Synthesizes tailored next-level concepts to bridge target seniority.
    4. Stores and returns fully formed roadmap data directly matching the frontend schema.
    """
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")

    user_data = user_doc.to_dict() or {}
    target_prefs = user_data.get("target_preferences", [{}])
    first_pref = target_prefs[0] if target_prefs else {}

    target_role = first_pref.get("role") or user_data.get("target_role") or "Software Engineer"
    target_company = first_pref.get("company") or "Technology"
    target_level = first_pref.get("level") or "Senior"

    skill_matrix = user_data.get("skill_matrix", {})
    known_skills = (
        user_data.get("parsed_profile", {}).get("skills", [])
        or user_data.get("parsed_skills", [])
        or user_data.get("skills", [])
    )

    prompt = f"""
Role: You are an Executive Tech Career Architect and Recruiter.

Candidate Telemetry:
- Known/Verified Skills: {json.dumps(known_skills)}
- Competency Matrix (Skill -> Score/10): {json.dumps(skill_matrix)}
- Target Role: {target_role}
- Target Seniority Level: {target_level}
- Target Company/Ecosystem: {target_company}

Your Goals:
1. "next_level_concepts":
   - For 2 to 4 technical domains in the candidate's matrix (especially where score < 8), identify:
     * domain: name of the domain.
     * current_proficiency: user's assessed score (1-10).
     * target_proficiency: 8-10.
     * what_user_knows: list of specific tools/patterns the candidate already commands.
     * target_concept: the exact enterprise production pattern required to bridge to {target_level}.
     * why_next_level: architectural justification.
     * implementation_blueprint: concise configuration/engineering steps.

2. "opportunities":
   - Search the live web for 3 to 4 REAL career opportunities, active postings, enterprise hackathons, or ecosystem challenges matching {target_role} / {target_company}.
   - Do NOT invent fake URLs or placeholders. Find actual links (Careers portals, Devpost, LinkedIn, official tech blogs).
   - Each opportunity must contain:
     * title: Job opening or challenge name.
     * company: Hiring company or hosting organization.
     * level: e.g. "L3", "Senior", "Hackathon", "Open Bounty".
     * match_percentage: Integer (55 to 95) calculated from their skills vs requirements.
     * compensation_range: Salary range or prize pool (e.g. "$160,000 - $210,000" or "$25,000 in Prizes").
     * unlocked_by: List of actual skills the candidate has that qualify them for this.
     * blocking_gaps: List of specific technical gaps they must master.
     * strategic_advice: High-signal advice on how to pass screening or submit a winning project.
     * url: Direct link to the listing or announcement.

Return ONLY a valid JSON object matching this structure:
{{
  "next_level_concepts": [
    {{
      "domain": "...",
      "current_proficiency": 4,
      "target_proficiency": 8,
      "what_user_knows": ["..."],
      "target_concept": "...",
      "why_next_level": "...",
      "implementation_blueprint": "..."
    }}
  ],
  "opportunities": [
    {{
      "title": "...",
      "company": "...",
      "level": "...",
      "match_percentage": 82,
      "compensation_range": "...",
      "unlocked_by": ["..."],
      "blocking_gaps": ["..."],
      "strategic_advice": "...",
      "url": "..."
    }}
  ]
}}
"""

    payload = {"next_level_concepts": [], "opportunities": []}

    try:
        if ai_client:
            response = await ai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                )
            )

            raw_text = response.text or ""
            logger.info(f"Gemini raw response text: {raw_text[:300]}...")

            # Extract clean JSON block even if preceded by explanations or citations
            json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if json_match:
                payload = json.loads(json_match.group(1))
            elif raw_text.strip():
                clean_text = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                payload = json.loads(clean_text)

    except Exception as exc:
        logger.error(f"Radar generation failed for user {user_id}: {exc}", exc_info=True)

    # Clean empty lists/keys if Gemini failed or partially returned
    concepts = payload.get("next_level_concepts", [])
    opportunities = payload.get("opportunities", [])

    # Persist directly into the Firestore doc that OpportunityRadarScreen listens to
    roadmap_doc_ref = user_ref.collection("horizon_analytics").document("next_level_roadmap")
    roadmap_doc_ref.set({
        "next_level_concepts": concepts,
        "opportunities": opportunities,
        "target_role": target_role,
        "target_level": target_level,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    return {
        "status": "success",
        "next_level_concepts": concepts,
        "opportunities": opportunities,
    }


@router.post("/generate/{user_id}", status_code=status.HTTP_200_OK)
async def generate_user_recommendations(user_id: str):
    """Triggers hybrid discovery (Gemini Grounded Docs + YouTube Videos) for skill gaps.
    
    Fails fast with descriptive HTTP status codes if any step fails.
    """
    try:
        # 1. Fetch User Profile
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User profile '{user_id}' does not exist."
            )

        user_data = user_doc.to_dict() or {}

        # Validate Skill Matrix
        skill_matrix = user_data.get("skill_matrix")
        if not skill_matrix or not isinstance(skill_matrix, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="User profile is missing a valid 'skill_matrix' dictionary."
            )

        # Validate Target Preferences
        target_prefs = user_data.get("target_preferences", [])
        if not target_prefs or not target_prefs[0].get("role"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="User profile is missing a target role in 'target_preferences'."
            )
        target_role = target_prefs[0]["role"]

        # 2. Identify the Lowest Scoring Skill Domains
        gaps = sorted(
            [
                (k, v.get("score", 0) if isinstance(v, dict) else 0) 
                for k, v in skill_matrix.items()
            ],
            key=lambda x: x[1]
        )
        if not gaps:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Skill matrix contains no evaluable skill scores."
            )

        logger.info(f"gaps: {gaps}")
        # 3. Hybrid Discovery: 1 Official Doc/Article + 1 YouTube Video per domain
        enriched_items = await run_discovery_pipeline(target_role, gaps)
        
        logger.info(f"enriched_items: {enriched_items}")

        # 5. Persist to Firestore->  removed .document("current_feed")
        user_ref.collection("recommendations").document("current_feed").set({
            "items": enriched_items,
            "target_role": target_role,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)

        user_ref.update({"recommendations": enriched_items})

        return {
            "status": "success",
            "user_id": user_id,
            "target_role": target_role,
            "total_items": len(enriched_items),
            "items": enriched_items
        }

    except HTTPException:
        # Re-raise explicit HTTP exceptions untouched
        raise
    except Exception as exc:
        logger.error(f"Unhandled error in generate_user_recommendations for user {user_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(exc)}"
        )