import logging
from typing import List, Dict, Any, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.config import YOUTUBE_API_KEY

logger = logging.getLogger(__name__)


class YouTubeDiscoveryService:
    def __init__(self):
        if not YOUTUBE_API_KEY:
            raise RuntimeError("Environment variable 'YOUTUBE_API_KEY' is not set.")
        self.client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    def search_videos_batch(
        self, 
        target_role: str, 
        domain_skills: List[Tuple[str, int]], 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes a SINGLE YouTube search call using a list of tuples: [('Domain', score), ...]
        Prioritizes lowest-scoring domains.
        """
        if not domain_skills:
            return []

        # Sort by score ascending to focus search on primary gaps
        sorted_domains = sorted(domain_skills, key=lambda x: x[1])
        priority_domains = " ".join([domain for domain, _ in sorted_domains[:3]])

        query = f"{target_role} {priority_domains} system design architecture tutorial"

        try:
            request = self.client.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results,
                relevanceLanguage="en",
                videoEmbeddable="true"
            )
            response = request.execute()

            items = []
            domain_names = [domain for domain, _ in domain_skills]

            for item in response.get("items", []):
                video_id = item["id"].get("videoId")
                if not video_id:
                    continue

                snippet = item["snippet"]
                video_title = snippet.get("title", "")
                video_desc = snippet.get("description", "")

                matched_domains = [
                    domain for domain in domain_names
                    if domain.lower() in video_title.lower() or domain.lower() in video_desc.lower()
                ]

                items.append({
                    "title": video_title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "source_type": "youtube",
                    "relevance_score": 0.92,
                    "snippet": video_desc,
                    "skill_domains": matched_domains if matched_domains else [sorted_domains[0][0]]
                })

            return items

        except Exception as exc:
            logger.error(f"YouTube search error: {exc}", exc_info=True)
            return []