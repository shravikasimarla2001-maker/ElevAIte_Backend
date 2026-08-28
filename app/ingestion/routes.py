# /api/v1/diagnostic endpoints
import json
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from google.cloud import firestore
from app.config import db
from app.ingestion.document_parser import parse_resume_pdf

router = APIRouter(prefix="/api/v1/onboard", tags=["Ingestion"])

class TargetPreference(BaseModel):
    company: str
    role: str


# # 1. Create a dummy resume PDF
# echo "%PDF-1.4 sample resume text for AI architect" > test_resume.pdf
# # 2. Run the onboarding request
# curl -X POST "http://localhost:8080/api/v1/onboard" \
#   -F "user_id=usr_1001" \
#   -F "email=shravikasimarla2001@gmail.com" \
#   -F 'target_preferences=[{"company": "Google", "role": "AI Solutions Architect"}]' \
#   -F "resume=@test_resume.pdf;type=application/pdf"

@router.post("", status_code=status.HTTP_201_CREATED)
async def onboard_user(
    user_id: str = Form(...),
    email: str = Form(...),
    target_preferences: str = Form(...),  # e.g., '[{"company":"Google","role":"AI Architect"}]'
    linkedin_url: Optional[str] = Form(None),
    github_username: Optional[str] = Form(None),
    resume: Optional[UploadFile] = File(None)
):
    try:
        raw_preferences = json.loads(target_preferences)
        if not isinstance(raw_preferences, list):
            raise ValueError("target_preferences must be a JSON array of objects.")

        parsed_preferences = [TargetPreference(**item).model_dump() for item in raw_preferences]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid target_preferences JSON: {str(e)}")

    parsed_resume_text = ""
    if resume:
        if not resume.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume file must be in PDF format."
            )

        file_bytes = await resume.read()
        parsed_resume_text = parse_resume_pdf(file_bytes, resume.content_type or "application/pdf")

    # Initial baseline skill matrix initialized at level 0/10 until diagnostic starts
    user_profile = {
        "user_id": user_id,
        "email": email,
        "linkedin_url": linkedin_url,
        "github_username": github_username,
        "target_preferences": parsed_preferences,
        "resume_text": parsed_resume_text,
        "skill_matrix": {
            "Core Architecture": {"score": 0, "status": "unassessed"},
            "System Design": {"score": 0, "status": "unassessed"},
            "Hands-on Implementation": {"score": 0, "status": "unassessed"},
            "Troubleshooting & Scale": {"score": 0, "status": "unassessed"}
        },
        "created_at": firestore.SERVER_TIMESTAMP
    }

    db.collection("users").document(user_id).set(user_profile, merge=True)
    return {"status": "success", "message": "Profile onboarded with initial skill matrix", "user_id": user_id}