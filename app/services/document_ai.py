import os
import re
import json
import logging
from google.cloud import documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google import genai
from google.genai import types

from app.config import (
    PROJECT_ID,
    DOCAI_PROJECT_ID,
    LOCATION,
    DOCAI_PROCESSOR_ID,
    VERTEX_REGION,
    GCS_BUCKET_NAME, 
    ai_client
)

logger = logging.getLogger("services.document_ai")
logging.basicConfig(level=logging.INFO)


# 1. Document AI Client
# doc_options = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com") if LOCATION else None
# docai_client = documentai.DocumentProcessorServiceClient(client_options=doc_options)
doc_options = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com") if LOCATION else None
docai_client = documentai.DocumentProcessorServiceClient(client_options=doc_options)

# 2. Cloud Storage Client
storage_client = storage.Client(project=PROJECT_ID)


def delete_old_resume_from_gcs(gcs_uri: str) -> None:
    """Deletes previous resume blob from Google Cloud Storage when replacing."""
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return
    try:
        # Format: gs://bucket-name/path/to/blob
        path_parts = gcs_uri.replace("gs://", "").split("/", 1)
        if len(path_parts) == 2:
            bucket_name, blob_name = path_parts
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            logger.info(f"Successfully deleted obsolete resume blob: {blob_name}")
    except NotFound:
        logger.warning(f"Old resume blob not found for deletion: {gcs_uri}")
    except Exception as e:
        logger.error(f"Failed to delete old resume from GCS ({gcs_uri}): {str(e)}")


def upload_pdf_to_gcs(user_id: str, file_bytes: bytes, filename: str) -> str:
    """Persists resume to GCS. Raises RuntimeError on storage failure."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"resumes/{user_id}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(file_bytes, content_type="application/pdf")
        return f"gs://{GCS_BUCKET_NAME}/{blob_path}"
    except GoogleAPICallError as e:
        logger.error(f"GCS Upload Failed: {e.message} (Code: {e.code})")
        raise RuntimeError(f"GCS bucket operation failed: {e.message}") from e
    except Exception as e:
        logger.error(f"Unexpected GCS storage error: {str(e)}")
        raise RuntimeError(f"Storage service encountered an internal failure: {str(e)}") from e
 
def extract_text_via_document_ai(file_bytes: bytes = None, file_name: str = "", uri: str = "") -> str:
    """Performs OCR layout text parsing via Document AI using either bytes or a GCS URI."""
    if not DOCAI_PROCESSOR_ID:
        raise ValueError("DOCAI_PROCESSOR_ID is missing or not set in backend configuration.")

    try:
        # 1. Derive the fully-qualified processor resource identifier
        resource_name = docai_client.processor_path(DOCAI_PROJECT_ID, LOCATION, DOCAI_PROCESSOR_ID)

        # 2. Choose input source: GCS URI (if provided) or In-memory file_bytes
        if uri:
            # For files stored in a Google Cloud Storage bucket
            gcs_document = documentai.GcsDocument(gcs_uri=uri, mime_type="application/pdf")
            request = documentai.ProcessRequest(
                name=resource_name,
                gcs_document=gcs_document
            )
            logger.info(f"Executing Document AI process request using GCS URI: {uri}")
        elif file_bytes:
            # For files passed directly as raw in-memory bytes
            raw_document = documentai.RawDocument(content=file_bytes, mime_type="application/pdf")
            request = documentai.ProcessRequest(
                name=resource_name,
                raw_document=raw_document
            )
            logger.info("Executing Document AI process request using raw bytes...")
        else:
            raise ValueError("Either file_bytes or a valid GCS uri must be provided.")

        # 3. Process document
        result = docai_client.process_document(request=request)
        extracted_text = result.document.text
        logger.info(f"Successfully extracted {len(extracted_text)} characters via Document AI.")

        if not extracted_text or not extracted_text.strip():
            raise ValueError("Document AI processed the document, but no textual characters were detected.")

        return extracted_text

    except GoogleAPICallError as e:
        logger.error(f"Document AI Process Error: {e.message}")
        raise RuntimeError(f"Document AI OCR call failed: {e.message}") from e

def _clean_and_parse_json(raw_text: str) -> dict:
    """Strict JSON parser that cleans markdown backticks."""
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned an empty or whitespace response.")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        logger.error(f"Invalid JSON emitted by LLM: {err}. Raw text:\n{raw_text}")
        raise ValueError(f"LLM output could not be parsed as valid JSON: {err.msg}") from err

    if not isinstance(data, dict):
        raise ValueError("LLM payload root must be a JSON object/dictionary.")
    
    if "dynamic_matrix" not in data or not isinstance(data["dynamic_matrix"], dict):
        raise ValueError("LLM response is missing the required 'dynamic_matrix' object.")

    if len(data["dynamic_matrix"]) == 0:
        raise ValueError("LLM returned an empty 'dynamic_matrix'. Could not calibrate domains.")

    return data


async def extract_dynamic_skill_matrix(raw_text: str, target_role: str, target_company: str = "Google") -> dict:
    """Extracts dynamic skills & domain matrix strictly via Vertex AI."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot extract skills: OCR text input is completely empty.")

    prompt = f"""
Role (Persona):
Act as an elite Technical Talent Architect and Engineering Lead at "{target_company}".

Context:
A candidate has uploaded a new resume to their career assistant profile ("ElevAIte") targeting the role: "{target_role}". You need to extract their verified baseline technical abilities, eliminate obsolete information, and establish a dynamic domain competency map.

Task:
Analyze the OCR layout text from the candidate's resume and perform the following:
1. Extract all verified technical skills, libraries, databases, and frameworks into a flat array named "parsed_skills".
2. Extract total years of professional experience as an integer into "years_experience".
3. Extract verified industry credentials and certifications into "certifications".
4. Derive 4 to 6 dynamic competency domain names based directly on what the candidate has built (e.g., "Distributed Systems", "Cloud & Serverless", "LLM Pipelines", "API Architecture").
5. For each domain, establish a baseline proficiency score (1 to 10), provide concise evidence feedback, and mark "verified_via" as "resume_ocr".

Constraints:
- Do not use generic domain names like "General Skills" or "Other".
- Do not hallucinate skills or certifications not grounded in the text.
- Do not return markdown explanations outside the specified JSON schema.
- All scores must be integers between 1 and 10.

Format:
Return ONLY a valid JSON object matching this exact schema:
{{
  "parsed_skills": ["Skill 1", "Skill 2"],
  "years_experience": 4,
  "certifications": ["Cert 1"],
  "dynamic_matrix": {{
    "Cloud Architecture": {{
      "score": 6,
      "last_feedback": "Demonstrated production deployment of containerized services.",
      "verified_via": "resume_ocr"
    }}
  }}
}}

Resume Text:
{raw_text[:8000]}

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
        return _clean_and_parse_json(response.text)
    except Exception as e:
        logger.error(f"Vertex AI Gemini extraction failure: {str(e)}")
        raise RuntimeError(f"Vertex AI Gemini extraction service error: {str(e)}") from e


async def recalibrate_existing_skills(
    existing_skills: list, 
    existing_matrix: dict, 
    new_target_role: str, 
    target_company: str = "General Tech"
) -> dict:
    """
    Dynamically recalculates domain scores and feedback against a new target role
    without re-parsing the PDF resume.
    """
    prompt = f"""
Role (Persona):
Act as a Principal Staff Engineer and Technical Hiring Bar Raiser at "{target_company}".

Context:
A candidate using "ElevAIte" has changed their career trajectory to a new target role: "{new_target_role}" at "{target_company}". Their existing verified technical skills and current competency domains are already established in the database, but their domain proficiency scores and gap alignment need immediate recalibration against the strict bar of this new target role without re-parsing their resume.

Task:
Evaluate the candidate's existing skills against the expectations of "{new_target_role}" at "{target_company}":
1. Re-evaluate every domain in "existing_matrix" against the technical rigor, depth, and standards required for "{new_target_role}".
2. Adjust each domain "score" up or down (1 to 10 scale) to reflect readiness for this specific role.
3. Write a target-specific "last_feedback" statement explaining the candidate's strengths or priority gaps relative to "{new_target_role}".
4. Set "verified_via" to "target_recalibration".

Constraints:
- Do NOT invent new unverified skills that are not supported by the candidate's existing skill list.
- Keep domain feedback actionable, technical, and concise (under 25 words per domain).
- Do NOT wrap the JSON in conversational filler or introductory/concluding text.

Format:
Return ONLY a valid JSON object matching this exact schema:
{{
  "dynamic_matrix": {{
    "<Domain Name>": {{
      "score": 7,
      "last_feedback": "Aligned with {new_target_role} expectations; priority gap in distributed state synchronization.",
      "verified_via": "target_recalibration"
    }}
  }}
}}

Existing Verified Skills:
{json.dumps(existing_skills)}

Current Matrix to Recalibrate:
{json.dumps(existing_matrix)}
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
        data = _clean_and_parse_json(response.text)
        logger.info(f"data:  {data}")
        return data.get("dynamic_matrix", {})
    except Exception as e:
        logger.error(f"Vertex AI skill recalibration failed: {str(e)}")
        raise RuntimeError(f"Failed to recalibrate skills for target role: {str(e)}") from e