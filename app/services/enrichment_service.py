import os
import logging
from typing import List, Dict, Any
from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from google import genai
from app.config import ai_client
from google.genai import types


logger = logging.getLogger(__name__)


class SummaryPayload(BaseModel):
    summary_points: List[str] = Field(..., min_items=3, max_items=3, description="3 bullet points outlining key takeaways")
    audio_briefing_text: str = Field(..., description="A 2-sentence spoken briefing script")


class RecommendationEnrichmentService:
    def __init__(self):
        if not ai_client:
            raise RuntimeError("Environment variable 'GEMINI_API_KEY' is not set.")

    async def generate_three_point_summary(
        self, title: str, snippet: str, skill_domains: List[str]
    ) -> Dict[str, Any]:
        """Generates 3 bullet points and an audio script using Gemini structured output."""
        prompt = (
            f"Analyze this recommended learning resource:\n"
            f"Title: {title}\n"
            f"Domains: {', '.join(skill_domains)}\n"
            f"Context: {snippet}\n\n"
            f"Provide exactly 3 concise technical takeaway points and a 2-sentence conversational audio briefing script."
        )

        try:
            res = await ai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": SummaryPayload,
                }
            )
            parsed = SummaryPayload.model_validate_json(res.text)
            return parsed.model_dump()

        except Exception as exc:
            logger.error(f"Enrichment summarization failed for '{title}': {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to generate summary payload via AI client: {str(exc)}"
            )