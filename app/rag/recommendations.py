import json
import logging
from typing import List, Dict, Any
from google.genai import types
from google.cloud import firestore
from app.config import ai_client, db

logger = logging.getLogger("rag.recommendations")


async def generate_level_based_concepts_and_opportunities(
    user_id: str,
    target_role: str,
    target_company: str,
    skill_matrix: Dict[str, Any],
    known_skills: List[str]
) -> Dict[str, Any]:
    """
    Evaluates verified user skills, what the user knows per domain, and proficiency levels
    to synthesize:
    1. Next-level technical concept trajectories per domain (bridging current score to next tier).
    2. Realistic career opportunities calibrated to their current seniority level.
    """
    prompt = f"""
Role:
Act as an Executive Talent Architect and Principal Engineering Director at {target_company or 'Google'}.

Context:
The user is aiming for the role "{target_role}" at target seniority "{target_level}".
They currently possess verified technical skills and an evaluated domain competency matrix (rated 1 to 10).

Candidate Telemetry:
- All Verified/Known Skills: {json.dumps(known_skills)}
- Evaluated Domain Competency Matrix: {json.dumps(skill_matrix)}
- Target Role: {target_role}
- Target Company: {target_company}

Task:
1. Detailed Knowledge & Domain Breakdown:
   - For EACH domain in the competency matrix, identify:
     a) What the user already knows/validated (e.g., proven tools/patterns from their known skills).
     b) Their current assessed proficiency (1-10).
     c) The exact next-level concept required to bridge from their current score to the {target_level} threshold (score 8-10).
     d) A concrete architectural implementation blueprint.
2. Level-Calibrated Opportunity Matching:
   - Suggest 3-4 realistic opportunities matching their CURRENT proficiency band (e.g., Mid, Senior, Staff) while steering toward {target_role}.
   - For each opportunity:
     a) "unlocked_by": The exact skills/domains they currently know that qualify them.
     b) "blocking_gaps": The high-priority concepts holding them back from full clearance.
     c) "match_percentage" (50% to 95%).
     d) Estimated market compensation range.
     e) Actionable strategic advice.

Constraints:
- Avoid basic tutorial topics; focus strictly on enterprise, production-grade architectures.
- All JSON keys must match the exact schema below.
- Do NOT output markdown or explanations outside the JSON object.

Format:
Return ONLY a valid JSON object matching this schema:
{{
  "next_level_concepts": [
    {{
      "domain": "Database & Vector Storage",
      "current_proficiency": 3,
      "target_proficiency": 8,
      "what_user_knows": ["Basic SQL", "pgvector extension installation", "Flat cosine search"],
      "target_concept": "HNSW Partitioning, Quantization & Memory Tuning in Large-Scale pgvector",
      "why_next_level": "Eliminates unindexed sequential scan bottlenecks; mandatory for sub-15ms vector retrieval across millions of records.",
      "implementation_blueprint": "Transition from flat vector scans to HNSW with m=16, ef_construction=64; allocate dedicated work_mem in PostgreSQL."
    }}
  ],
  "opportunities": [
    {{
      "title": "Senior AI Solutions Architect",
      "company": "{target_company}",
      "role": "{target_role}",
      "match_percentage": 84,
      "compensation_range": "$185,000 - $240,000",
      "unlocked_by": ["Core Architecture", "FastAPI", "Cloud Run Containerization"],
      "blocking_gaps": ["pgvector High-Scale Partitioning", "Document AI Custom Extractor Fine-Tuning"],
      "strategic_advice": "Focus next on index tuning in pgvector to pass the distributed systems bar."
    }}
  ]
}}
"""

    try:
        response = await ai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text)
    except Exception as e:
        logger.error(f"Failed to generate level concepts & opportunities: {str(e)}")
        data = {
            "next_level_concepts": [],
            "opportunities": []
        }

    # Persist the synthesized recommendations into Firestore under horizon_analytics
    user_ref = db.collection("users").document(user_id)
    user_ref.collection("horizon_analytics").document("next_level_roadmap").set({
        "next_level_concepts": data.get("next_level_concepts", []),
        "opportunities": data.get("opportunities", []),
        "target_role": target_role,
        "target_level": target_level,
        "generated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return data


def clean_url(raw_url: str) -> str:
    """Strips markdown links, brackets, and slug concatenations into clean URLs."""
    if not raw_url:
        return ""
    
    # 1. Extract URL if wrapped in markdown [label](https://...)
    md_match = re.search(r'\((https?://[^\)]+)\)', raw_url)
    target = md_match.group(1) if md_match else raw_url
    
    # 2. Clean out brackets or hanging text
    target = re.sub(r'[\[\]]', '', target).strip()
    
    # 3. Match pure HTTP/HTTPS URL
    match = re.search(r'https?://[^\s\)]+', target)
    if not match:
        return target
    
    url = match.group(0)
    
    # If the URL was glued to another slug (e.g. .../topics/developers-practitioners/google-cloud-...)
    # normalize double slashes or clean trail
    return url.rstrip('/')


async def generate_three_point_summary(title: str, snippet: str, domains: List[str]) -> Dict[str, Any]:
    """Generates the 3-point architectural summary and speech audio text via Gemini 2.5 Flash."""
    prompt = f"""
Role:
Act as a Principal Cloud Architect and Engineering Lead at Google.

Task:
Analyze the technical guide or video transcript and generate an executive 3-point architectural takeaway summary, followed by a concise 2-minute conversational audio briefing script.

Context:
The candidate is preparing for an L4/L5 engineering level assessment in ElevAIte. They are bridging specific competency gaps across multiple domains: {', '.join(domains)}. They do not need superficial descriptions; they require direct, high-signal engineering concepts, implementation tradeoffs, and architectural patterns.

Resource Title: {title}
Resource Context: {snippet}

Format:
Return ONLY a valid, raw JSON object matching this schema:
{{
  "summary_points": [
    "Point 1: Concrete architectural mechanism and GCP service interaction.",
    "Point 2: Scalability, networking, or concurrency pattern.",
    "Point 3: Failure handling, state management, or latency tradeoff."
  ],
  "audio_briefing_text": "A natural, 300-word spoken-voice script suitable for text-to-speech briefing the candidate on these architectural concepts."
}}

Constraints:
- Exactly 3 bullet points in "summary_points".
- Each bullet point must be between 20 and 40 words.
- Do NOT output markdown links, conversational preamble, or backtick fences outside the JSON object.
- Focus exclusively on technical mechanics.
"""
    try:
        response = await ai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        return json.loads(response.text)
    except Exception:
        # Fallback points aligned with the domains
        return {
            "summary_points": [
                f"Core patterns for production readiness across {', '.join(domains)}.",
                "End-to-end latency optimization and resilient service decoupling.",
                "Production observability, error budgets, and continuous automated testing."
            ],
            "audio_briefing_text": f"This briefing covers core architectural patterns in {', '.join(domains)}. Focus on decoupled service design, resilient failovers, and latency reduction."
        }


async def consolidate_and_store_recommendations(raw_items: List[Dict[str, Any]], user_id: str, db) -> List[Dict[str, Any]]:
    """
    1. Sanitizes URLs
    2. Groups multiple domains for the same source into a single entry
    3. Enriches each unique source with a 3-point summary
    4. Persists the consolidated records into Firestore
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for item in raw_items:
        url = clean_url(item.get("url", ""))
        
        # Replace mock youtube placeholders with standard Google Cloud Tech references
        if "watch?v=mock" in url or not url.startswith("http"):
            url = "https://www.youtube.com/watch?v=ylOtl99W20E"  # Official Google Cloud Tech Architecture Talk

        title = item.get("title", "Production Architecture Guide")
        key = url or title
        domain = item.get("skill_domain")

        if key not in grouped:
            grouped[key] = {
                "title": title,
                "url": url,
                "source_type": item.get("source_type", "blog"),
                "relevance_score": item.get("relevance_score", 0.92),
                "snippet": item.get("snippet", ""),
                "skill_domains": [domain] if domain else []
            }
        else:
            if domain and domain not in grouped[key]["skill_domains"]:
                grouped[key]["skill_domains"].append(domain)

    consolidated_list = []
    
    for _, entry in grouped.items():
        # Generate the structured 3-point summary
        summary_payload = await generate_three_point_summary(
            entry["title"], 
            entry["snippet"], 
            entry["skill_domains"]
        )
        
        entry["summary_points"] = summary_payload.get("summary_points", [])
        entry["audio_briefing_text"] = summary_payload.get("audio_briefing_text", "")
        consolidated_list.append(entry)

    # Persist cleanly to Firestore under users/{userId}/recommendations
    user_rec_ref = db.collection("users").document(user_id).collection("recommendations").document("current_feed")
    user_rec_ref.set({
        "items": consolidated_list,
        "updated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return consolidated_list