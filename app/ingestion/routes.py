import json
import logging
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
# pyrefly: ignore [missing-import]
from google.api_core.exceptions import GoogleAPICallError

from app.config import db
from app.services.document_ai import (
    extract_text_via_document_ai,
    extract_dynamic_skill_matrix,
    recalibrate_existing_skills
)

logger = logging.getLogger("ingestion.routes")
router = APIRouter(prefix="/api/v1")


@router.post("/onboard", status_code=status.HTTP_200_OK)
async def onboard_user(
    user_id: str = Form(...),
    email: str = Form(""),
    target_preferences: str = Form("[]"),
    linkedin_url: str = Form(""),
    github_username: str = Form(""),
    resume: UploadFile = File(None)
):
    # 1. Validate Form & User ID
    if not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required."
        )

    # 2. Parse Target Preferences JSON
    try:
        parsed_targets = json.loads(target_preferences) if target_preferences else []
        if not isinstance(parsed_targets, list):
            raise ValueError("Target preferences must be a valid JSON array.")
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed target_preferences JSON: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    target_role = (
        parsed_targets[0].get("role", "AI Solutions Architect")
        if parsed_targets and len(parsed_targets) > 0 and isinstance(parsed_targets[0], dict)
        else "AI Solutions Architect"
    )
    target_company = (
        parsed_targets[0].get("company", "General Tech")
        if parsed_targets and len(parsed_targets) > 0 and isinstance(parsed_targets[0], dict)
        else "General Tech"
    )

    # 3. Fetch Existing Profile from Firestore
    try:
        user_doc_ref = db.collection("users").document(user_id)
        existing_doc = user_doc_ref.get()
        existing_data = existing_doc.to_dict() if existing_doc.exists else {}
    except GoogleAPICallError as e:
        logger.error(f"Firestore Read Failure: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database retrieval error: {e.message}"
        )

    now = datetime.now(timezone.utc).isoformat()
    old_target_role = ""
    if existing_data.get("target_preferences") and len(existing_data["target_preferences"]) > 0:
        old_target_role = existing_data["target_preferences"][0].get("role", "")

    target_changed = bool(target_role and old_target_role and target_role.strip().lower() != old_target_role.strip().lower())

    # Pre-populate variables from existing profile
    update_payload = {
        "user_id": user_id,
        "email": email or existing_data.get("email", ""),
        "target_preferences": parsed_targets or existing_data.get("target_preferences", []),
        "linkedin_url": linkedin_url or existing_data.get("linkedin_url", ""),
        "github_username": github_username or existing_data.get("github_username", ""),
        "updated_at": now
    }

    # =========================================================================
    # CASE 1: A NEW RESUME IS UPLOADED (Wipe old resume & skills, create fresh)
    # =========================================================================
    if resume and resume.filename:
        if not resume.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF documents are supported for resume upload."
            )

        file_bytes = await resume.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty (0 bytes)."
            )

        try:
            # 1. Document AI OCR layout extraction
            ocr_text = extract_text_via_document_ai(file_bytes)

            # 4. Generate fresh skills & dynamic domain matrix via Vertex AI
            extracted_data = await extract_dynamic_skill_matrix(ocr_text, target_role)

            new_skills = extracted_data.get("parsed_skills", [])
            new_certifications = extracted_data.get("certifications", [])
            new_years_experience = extracted_data.get("years_experience", 0)
            new_dynamic_matrix = extracted_data.get("dynamic_matrix", {})

            # 5. Completely overwrite previous resume & skills data
            update_payload["resume"] = {
                "filename": resume.filename,
                "raw_ocr_text": ocr_text,
                "uploaded_at": now
            }
            update_payload["resume_filename"] = resume.filename
            update_payload["parsed_skills"] = new_skills
            update_payload["certifications"] = new_certifications
            update_payload["years_experience"] = new_years_experience
            update_payload["skill_matrix"] = new_dynamic_matrix

            # 6. Overwrite the isolated subcollection snapshot
            matrix_subdoc_ref = user_doc_ref.collection("metadata").document("skill_matrix")
            matrix_subdoc_ref.set({
                "domains": new_dynamic_matrix,
                "last_updated": now,
                "source": "resume_ocr"
            })

            # Save full replacement to Firestore
            user_doc_ref.set(update_payload, merge=True)

            return {
                "status": "success",
                "action": "resume_replaced",
                "user_id": user_id,
                "resume_saved": resume.filename,
                "extracted_skills": new_skills,
                "dynamic_matrix": new_dynamic_matrix,
                "domains_created": list(new_dynamic_matrix.keys()),
                "linkedin_url": update_payload["linkedin_url"],
                "github_username": update_payload["github_username"]
            }

        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    

    # =========================================================================
    # CASE 2: NO NEW RESUME, BUT TARGET ROLE WAS UPDATED (Recalibrate skills)
    # =========================================================================
    elif target_changed:
        logger.info(f"Target role changed from '{old_target_role}' to '{target_role}'. Recalibrating...")
        existing_skills = existing_data.get("parsed_skills", [])
        existing_matrix = existing_data.get("skill_matrix", {})

        recalibrated_matrix = await recalibrate_existing_skills(
            existing_skills=existing_skills,
            existing_matrix=existing_matrix,
            new_target_role=target_role,
            target_company=target_company
        )

        # Retain all resume references
        if "resume" in existing_data:
            update_payload["resume"] = existing_data["resume"]
        if "resume_filename" in existing_data:
            update_payload["resume_filename"] = existing_data["resume_filename"]

        update_payload["parsed_skills"] = existing_skills
        update_payload["certifications"] = existing_data.get("certifications", [])
        update_payload["years_experience"] = existing_data.get("years_experience", 0)
        update_payload["skill_matrix"] = recalibrated_matrix

        # Update metadata subcollection
        matrix_subdoc_ref = user_doc_ref.collection("metadata").document("skill_matrix")
        matrix_subdoc_ref.set({
            "domains": recalibrated_matrix,
            "last_updated": now,
            "source": "target_recalibration"
        }, merge=True)

        user_doc_ref.set(update_payload, merge=True)

        return {
            "status": "success",
            "action": "target_recalibrated",
            "user_id": user_id,
            "resume_saved": existing_data.get("resume_filename", ""),
            "extracted_skills": existing_skills,
            "dynamic_matrix": recalibrated_matrix,
            "domains_created": list(recalibrated_matrix.keys()),
            "linkedin_url": update_payload["linkedin_url"],
            "github_username": update_payload["github_username"]
        }

    # =========================================================================
    # CASE 3: MINOR EDIT (Social links, company name only)
    # =========================================================================
    else:
        if "resume" in existing_data:
            update_payload["resume"] = existing_data["resume"]
        if "resume_filename" in existing_data:
            update_payload["resume_filename"] = existing_data["resume_filename"]

        update_payload["parsed_skills"] = existing_data.get("parsed_skills", [])
        update_payload["certifications"] = existing_data.get("certifications", [])
        update_payload["years_experience"] = existing_data.get("years_experience", 0)
        update_payload["skill_matrix"] = existing_data.get("skill_matrix", {})

        user_doc_ref.set(update_payload, merge=True)

        return {
            "status": "success",
            "action": "profile_updated",
            "user_id": user_id,
            "resume_saved": update_payload.get("resume_filename", ""),
            "extracted_skills": update_payload["parsed_skills"],
            "dynamic_matrix": update_payload["skill_matrix"],
            "domains_created": list(update_payload["skill_matrix"].keys()),
            "linkedin_url": update_payload["linkedin_url"],
            "github_username": update_payload["github_username"]
        }