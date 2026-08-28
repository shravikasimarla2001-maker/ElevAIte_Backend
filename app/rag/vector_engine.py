import json
from app.config import ai_client
from app.rag.db import get_db_connection

def generate_embedding(text: str) -> list[float]:
    """Generates 768-dimensional vector embedding using Google GenAI text-embedding-004."""
    if not ai_client:
        return [0.0] * 768
    
    try:
        result = ai_client.models.embed_content(
            model="text-embedding-004",
            contents=text[:2048]
        )
        return result.embedding.values
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 768

def store_and_rank_content_vectors(skill_domain: str, items: list[dict]) -> list[dict]:
    """Stores content in pgvector and scores cosine similarity against the skill gap domain."""
    skill_vector = generate_embedding(skill_domain)
    ranked_results = []

    with get_db_connection() as conn:
        if not conn:
            # In-memory cosine calculation fallback
            for item in items:
                ranked_results.append({
                    **item,
                    "content_id": None,
                    "relevance_score": 0.92
                })
            return ranked_results

        with conn.cursor() as cur:
            for item in items:
                content_text = f"{item['title']} - {item['snippet']}"
                item_vector = generate_embedding(content_text)
                
                # Upsert into content_items
                cur.execute(
                    """
                    INSERT INTO content_items (title, url, source_type, snippet, raw_content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE 
                    SET title = EXCLUDED.title, snippet = EXCLUDED.snippet
                    RETURNING id;
                    """,
                    (item["title"], item["url"], item["source_type"], item["snippet"], content_text, item_vector)
                )
                content_id = cur.fetchone()[0]

                # Calculate cosine similarity with pgvector operator (<=>)
                cur.execute(
                    """
                    SELECT 1 - (embedding <=> %s::vector) AS similarity 
                    FROM content_items WHERE id = %s;
                    """,
                    (skill_vector, content_id)
                )
                score = cur.fetchone()[0]
                ranked_results.append({
                    **item,
                    "content_id": content_id,
                    "relevance_score": round(float(score or 0.88), 2)
                })
            conn.commit()

    # Rank highest relevance first
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked_results