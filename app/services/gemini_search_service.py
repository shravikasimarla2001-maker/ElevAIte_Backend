import logging
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai import types
from app.config import ai_client

logger = logging.getLogger(__name__)


class GeminiGroundedSearchService:
    def __init__(self):
        if not ai_client:
            raise RuntimeError("Environment variable 'GEMINI_API_KEY' is not set.")

    async def search_articles_batch(
        self, 
        target_role: str, 
        domain_skills: List[Tuple[str, int]]
    ) -> List[Dict[str, Any]]:
        """
        Executes a SINGLE Gemini Grounded Search call accepting a list of tuples: [('Domain', score), ...]
        """
        if not domain_skills:
            return []

        skills_summary = "\n".join(
            [f"- {domain} (Current Skill: {score}/10)" for domain, score in domain_skills]
        )

        prompt = (
            f"You are a principal tech lead curating learning materials for a candidate aiming for the role: '{target_role}'.\n"
            f"Here are their target domains and current skill assessment levels:\n"
            f"{skills_summary}\n\n"
            f"For these domain, find 3 to 4 authoritative, production-grade technical guides, architectural blueprints, "
            f"or official engineering articles tailored to bridge their skill gaps."
        )

        try:
            response = await ai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2,
                    max_output_tokens=1024
                )
            )

            grounding_chunks = (
                response.candidates[0].grounding_metadata.grounding_chunks
                if response.candidates and response.candidates[0].grounding_metadata
                else []
            )

            if not grounding_chunks:
                return []

            items = []
            seen_urls = set()
            domain_names = [domain for domain, _ in domain_skills]

            for chunk in grounding_chunks:
                if chunk.web and chunk.web.uri:
                    link = chunk.web.uri
                    if link in seen_urls:
                        continue
                    seen_urls.add(link)

                    title = chunk.web.title or "Architecture Reference"
                    matched_domains = [
                        domain for domain in domain_names 
                        if domain.lower() in title.lower() or domain.lower() in link.lower()
                    ]

                    items.append({
                        "title": title,
                        "url": link,
                        "source_type": "official_doc" if any(x in link for x in ("cloud.google", "aws.amazon", "learn.microsoft", "github.com", "martinfowler")) else "blog",
                        "relevance_score": 0.95,
                        "snippet": response.text[:300].strip() if response.text else "Architecture reference.",
                        "skill_domains": matched_domains if matched_domains else domain_names
                    })

            return items

        except Exception as exc:
            logger.error(f"Gemini Grounded Search error: {exc}", exc_info=True)
            return []