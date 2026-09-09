import json
import logging
import re
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from app.config import db, ai_client
from app.rag.discovery import search_google_custom

logger = logging.getLogger("rag.routes")
router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations & Horizon"])


def clean_url(raw_url: str) -> str:
    """Strips markdown syntax, brackets, and slug concatenations."""
    if not raw_url:
        return ""
    md_match = re.search(r'\((https?://[^\)]+)\)', raw_url)
    target = md_match.group(1) if md_match else raw_url
    target = re.sub(r'[\[\]]', '', target).strip()
    match = re.search(r'https?://[^\s\)]+', target)
    return match.group(0).rstrip('/') if match else target


async def generate_three_point_summary(title: str, snippet: str, domains: List[str]) -> Dict[str, Any]:
    """Generates 3-point architectural takeaway summary using Gemini."""
    prompt = f"""
Analyze this technical guide or talk for an L4/L5 engineering candidate bridging gaps in: {', '.join(domains)}.
Resource: {title}
Context: {snippet}

Return ONLY valid JSON matching this schema:
{{
  "summary_points": [
    "Point 1: Concrete architectural mechanism and cloud service interaction (25-40 words).",
    "Point 2: Scalability, networking, or concurrency pattern (25-40 words).",
    "Point 3: Failure handling, state management, or latency tradeoff (25-40 words)."
  ],
  "audio_briefing_text": "A concise 200-word spoken audio briefing summarizing the architecture tradeoffs."
}}
"""
    try:
        if ai_client:
            resp = await ai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.2}
            )
            return json.loads(resp.text)
    except Exception as e:
        logger.warning(f"Summary generation fallback triggered: {e}")

    return {
        "summary_points": [
            f"Core architecture patterns for high-availability production workloads across {', '.join(domains)}.",
            "End-to-end latency reduction through connection pooling, caching, and decoupled service layers.",
            "Resilience patterns including exponential backoffs, dead-letter queues, and error budgets."
        ],
        "audio_briefing_text": f"This briefing covers core architectural patterns for {', '.join(domains)}. Focus on decoupled service design, resilient failovers, and latency reduction."
    }


# Accepts BOTH GET and POST to prevent 405 errors from frontend fetch/navigator calls
@router.api_route("/radar/discover/{user_id}", methods=["GET", "POST"], status_code=status.HTTP_200_OK)
async def discover_market_opportunities(user_id: str):
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict() or {}
    target_prefs = user_data.get("target_preferences", [{}])
    first_pref = target_prefs[0] if target_prefs else {}
    target_role = first_pref.get("role", "AI Solutions Architect")
    target_company = first_pref.get("company", "Google")

    # Ingest active opportunities from search indexers
    opportunity_queries = [
        f"{target_company} {target_role} careers openings apply",
        f"Google Cloud {target_role} hackathon challenge devpost 2026",
        f"GDG Cloud meetup {target_role} community"
    ]
    
    discovered_ops = []
    for q in opportunity_queries:
        try:
            results = await search_google_custom(q)
            for res in results[:1]:
                op_type = "Job Opening" if "careers" in q else ("Hackathon" if "hackathon" in q else "Meetup")
                discovered_ops.append({
                    "title": res.get("title", f"{target_company} Opportunity"),
                    "url": clean_url(res.get("url", "")),  # Cleaned markdown formatting
                    "snippet": res.get("snippet", ""),
                    "type": op_type,
                    "target_company": target_company,
                    "fit_score": 0.88 if op_type == "Job Opening" else 0.94
                })
        except Exception as search_err:
            logger.error(f"Error querying custom search for '{q}': {search_err}")

    # 1. Update legacy field on the root user doc
    user_ref.update({
        "radar_opportunities": discovered_ops
    })

    # 2. Persist to horizon_analytics/next_level_roadmap so the React snapshot listener catches it immediately
    user_ref.collection("horizon_analytics").document("next_level_roadmap").set({
        "opportunities": discovered_ops,
        "last_discovered_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return {"status": "success", "opportunities": discovered_ops}


@router.post("/radar/discover/v2/{user_id}", status_code=status.HTTP_200_OK)
async def get_horizon_recommendations(user_id: str):
    """
    Evaluates verified user skills, domain proficiency, and what the user knows
    to synthesize next-level concepts and level-calibrated opportunities.
    """
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")

    user_data = user_doc.to_dict() or {}
    target_prefs = user_data.get("target_preferences", [{}])
    first_pref = target_prefs[0] if target_prefs else {}

    target_role = first_pref.get("role") or "AI Solutions Architect"
    target_company = first_pref.get("company") or "Google"
    target_level = first_pref.get("level") or "Senior / L5"

    skill_matrix = user_data.get("skill_matrix", {})
    known_skills = (
        user_data.get("parsed_profile", {}).get("skills", [])
        or user_data.get("parsed_skills", [])
        or user_data.get("skills", [])
    )

    prompt = f"""
Role:
Act as an Executive Talent Architect and Principal Engineering Director at {target_company}.

Context:
The user is targeting the role "{target_role}" at target level "{target_level}".
They currently possess verified technical skills and an evaluated domain competency matrix (rated 1 to 10).

Candidate Telemetry:
- All Verified/Known Skills: {json.dumps(known_skills)}
- Evaluated Domain Competency Matrix: {json.dumps(skill_matrix)}
- Target Role: {target_role}
- Target Company: {target_company}
- Target Level: {target_level}

Task:
1. Detailed Knowledge & Domain Breakdown:
   - For EACH domain present in the competency matrix (or general core domains if empty):
     a) "what_user_knows": Specific tools, frameworks, and patterns they already have verified.
     b) "current_proficiency": Integer 1-10.
     c) "target_proficiency": Integer (e.g., 8-10).
     d) "target_concept": The exact production-grade architectural concept to master next.
     e) "why_next_level": Architectural justification for {target_level}.
     f) "implementation_blueprint": Practical engineering steps or configuration notes.
2. Level-Calibrated Opportunity Matching:
   - Generate 3-4 realistic opportunities matching their current experience trajectory:
     a) "title", "company", "level".
     b) "match_percentage" (50 to 95%).
     c) "compensation_range".
     d) "unlocked_by": Array of skills they already possess qualifying them.
     e) "blocking_gaps": Concrete technical topics preventing clearance.
     f) "strategic_advice": Actionable priority advice.

Constraints:
- Return ONLY a valid JSON object matching the schema below.
- Do NOT output markdown fences or commentary outside the JSON.

Schema:
{{
  "next_level_concepts": [
    {{
      "domain": "Database & Vector Storage",
      "current_proficiency": 3,
      "target_proficiency": 8,
      "what_user_knows": ["PostgreSQL", "Basic Vector Extensions"],
      "target_concept": "HNSW vs IVFFlat Partitioning & Quantization in Large-Scale pgvector",
      "why_next_level": "Eliminates unindexed sequential scan bottlenecks; mandatory for sub-15ms vector retrieval across millions of dimensions.",
      "implementation_blueprint": "Transition from flat vector scans to HNSW with m=16, ef_construction=64; configure maintenance_work_mem."
    }}
  ],
  "opportunities": [
    {{
      "title": "Senior AI Solutions Architect",
      "company": "{target_company}",
      "level": "{target_level}",
      "match_percentage": 84,
      "compensation_range": "$185,000 - $245,000",
      "unlocked_by": ["FastAPI", "Cloud Run Containerization"],
      "blocking_gaps": ["pgvector High-Scale Partitioning", "Document AI Custom Extractor Fine-Tuning"],
      "strategic_advice": "Clear the vector database gap this sprint to qualify for technical screening rounds."
    }}
  ]
}}
"""
    try:
        if ai_client:
            response = await ai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.1}
            )
            payload = json.loads(response.text)
        else:
            payload = {"next_level_concepts": [], "opportunities": []}
    except Exception as err:
        logger.error(f"Failed to generate level concepts & opportunities: {err}")
        payload = {"next_level_concepts": [], "opportunities": []}

    # Persist generated suggestions to Firestore
    user_ref.collection("horizon_analytics").document("next_level_roadmap").set({
        "next_level_concepts": payload.get("next_level_concepts", []),
        "opportunities": payload.get("opportunities", []),
        "target_role": target_role,
        "target_level": target_level,
        "generated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return {
        "status": "success",
        "user_id": user_id,
        "target_role": target_role,
        "target_level": target_level,
        "next_level_concepts": payload.get("next_level_concepts", []),
        "opportunities": payload.get("opportunities", [])
    }


@router.post("/generate/{user_id}", status_code=status.HTTP_200_OK)
async def generate_user_recommendations(user_id: str):
    """Triggers RAG recommendation discovery with safe fallbacks."""
    try:
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_data = user_doc.to_dict() or {}
        skill_matrix = user_data.get("skill_matrix", {})
        target_prefs = user_data.get("target_preferences", [{}])
        target_role = target_prefs[0].get("role", "AI Solutions Architect") if target_prefs else "AI Solutions Architect"

        # Identify lowest scoring domains
        gaps = sorted(
            [(k, v.get("score", 0) if isinstance(v, dict) else 0) for k, v in skill_matrix.items()],
            key=lambda x: x[1]
        )
        target_domains = [g[0] for g in gaps[:2]] if gaps else ["Distributed Systems Concepts", "API Design & Integration"]

        raw_items = [
            {
                "title": f"Production Architecture Patterns for {target_domains[0]}",
                "url": f"https://cloud.google.com/blog/topics/developers-practitioners/google-cloud-{target_domains[0].lower().replace(' ', '-')}-architecture-guide",
                "source_type": "blog",
                "relevance_score": 0.94,
                "snippet": f"A comprehensive technical deep-dive into scalable, resilient cloud architectures for {target_domains[0]}.",
                "skill_domains": [target_domains[0]]
            },
            {
                "title": f"Deep Dive: Building Enterprise Systems with {target_domains[0]}",
                "url": "https://www.youtube.com/watch?v=ylOtl99W20E",
                "source_type": "youtube",
                "relevance_score": 0.91,
                "snippet": f"Hands-on architectural guide and code walk-through for enterprise {target_domains[0]}.",
                "skill_domains": [target_domains[0]]
            }
        ]

        if len(target_domains) > 1:
            raw_items.append({
                "title": f"High-Scale Implementation Strategies for {target_domains[1]}",
                "url": f"https://cloud.google.com/blog/topics/developers-practitioners/google-cloud-{target_domains[1].lower().replace(' ', '-')}-patterns",
                "source_type": "blog",
                "relevance_score": 0.89,
                "snippet": f"Production-grade blueprints, API quotas, and concurrency management for {target_domains[1]}.",
                "skill_domains": [target_domains[1]]
            })

        enriched_items = []
        for item in raw_items:
            item["url"] = clean_url(item["url"])
            summary_payload = await generate_three_point_summary(item["title"], item["snippet"], item["skill_domains"])
            item["summary_points"] = summary_payload.get("summary_points", [])
            item["audio_briefing_text"] = summary_payload.get("audio_briefing_text", "")
            enriched_items.append(item)

        user_ref.collection("recommendations").document("current_feed").set({
            "items": enriched_items,
            "target_role": target_role,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)

        user_ref.update({"recommendations": enriched_items})

        return {
            "status": "success",
            "user_id": user_id,
            "items": enriched_items
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in generate_user_recommendations: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))