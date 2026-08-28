from fastapi import APIRouter, HTTPException, status
from app.config import db
from app.rag.discovery import generate_search_queries_for_gaps, search_google_custom
from app.rag.vector_engine import store_and_rank_content_vectors
from app.rag.recommender import generate_personalized_takeaways, save_recommendations

router = APIRouter(prefix="/api/v1/recommendations", tags=["RAG Recommendations"])

@router.post("/generate/{user_id}", status_code=status.HTTP_200_OK)
async def generate_user_recommendations(user_id: str):
    """Triggers the complete RAG discovery and recommendation pipeline for a user."""
    # 1. Fetch user's current skill matrix from Firestore
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    user_data = user_doc.to_dict()
    skill_matrix = user_data.get("skill_matrix", {})
    target_prefs = user_data.get("target_preferences", [{"role": "AI Solutions Architect"}])
    target_role = target_prefs[0].get("role", "AI Solutions Architect") if target_prefs else "AI Solutions Architect"

    # 2. Identify top 2 weakest/unassessed skill domains
    gaps = []
    for domain, meta in skill_matrix.items():
        score = meta.get("score", 0)
        gaps.append((domain, score))
    
    # Sort by lowest score first
    gaps.sort(key=lambda x: x[1])
    target_gaps = gaps[:2] if gaps else [("Core Architecture", 4)]

    all_recommendations = []

    # 3. Discover and process content for each gap
    for domain, score in target_gaps:
        queries = await generate_search_queries_for_gaps(domain, score, target_role)
        discovered_content = []
        for q in queries:
            results = await search_google_custom(q)
            discovered_content.extend(results)

        # 4. Vector Embedding & Cosine Ranking via pgvector
        ranked = store_and_rank_content_vectors(domain, discovered_content)

        # 5. Gemini Personalized Takeaway Synthesis
        for item in ranked[:2]:
            item["takeaways"] = generate_personalized_takeaways(
                target_role, domain, score, item["title"], item["snippet"]
            )
            item["skill_domain"] = domain
            all_recommendations.append(item)

        # 6. Save to Database
        save_recommendations(user_id, target_role, domain, ranked[:2])

    # 7. Mirror to Firestore for instant real-time synchronization
    db.collection("users").document(user_id).update({
        "recommendations": all_recommendations
    })

    return {
        "status": "success",
        "user_id": user_id,
        "recommendations_count": len(all_recommendations),
        "recommendations": all_recommendations
    }

@router.get("/user/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_recommendations(user_id: str):
    """Fetches real-time recommendations stored in Firestore."""
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_doc.to_dict().get("recommendations", [])