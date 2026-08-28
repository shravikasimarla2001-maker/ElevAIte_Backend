# /api/v1/diagnostic endpoints
import uuid
from fastapi import APIRouter, Form, HTTPException, status
from google.cloud import firestore
from app.config import db, llm_model
from app.diagnostic.engine import generate_first_question, evaluate_and_generate_next

router = APIRouter(prefix="/api/v1/diagnostic", tags=["Diagnostic Interview"])

# curl -X POST "http://localhost:8080/api/v1/diagnostic/start" \
#   -F "user_id=usr_1001" \
#   -F "target_role=AI Solutions Architect"
@router.post("/start", status_code=status.HTTP_200_OK)
async def start_interview(user_id: str = Form(...), target_role: str = Form(...)):
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = user_doc.to_dict()
    resume_text = user_data.get("resume_text", "")
    skill_matrix = user_data.get("skill_matrix", {})

    q1_data = generate_first_question(target_role, resume_text, skill_matrix)

    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "target_role": target_role,
        "current_question_number": 1,
        "turns": [
            {
                "question_number": 1,
                "question": q1_data["question"],
                "skill_domain": q1_data.get("skill_domain", "Core Architecture"),
                "answer": None,
                "score": None,
                "feedback": None
            }
        ],
        "status": "in_progress",
        "created_at": firestore.SERVER_TIMESTAMP
    }
    
    # Save session
    db.collection("users").document(user_id)\
      .collection("diagnostic_sessions").document(session_id).set(session_data)

    return {
        "session_id": session_id,
        "question_number": 1,
        "question": q1_data["question"],
        "skill_domain": q1_data.get("skill_domain"),
        "total_questions": 5
    }


# curl -X POST "http://localhost:8080/api/v1/diagnostic/submit-turn" \
#   -F "user_id=usr_1001" \
#   -F "session_id=YOUR_SESSION_ID_FROM_STEP_2" \
#   -F "question_number=1" \
#   -F "user_answer=I use Cloud Run microservices communicating via gRPC streaming with Cloud Pub/Sub handling decoupled async events."

@router.post("/submit-turn", status_code=status.HTTP_200_OK)
async def submit_turn(
    user_id: str = Form(...),
    session_id: str = Form(...),
    question_number: int = Form(...),
    user_answer: str = Form(...)
):
    user_ref = db.collection("users").document(user_id)
    session_ref = user_ref.collection("diagnostic_sessions").document(session_id)
    
    session_doc = session_ref.get()
    user_doc = user_ref.get()
    if not session_doc.exists or not user_doc.exists:
        raise HTTPException(status_code=404, detail="Session or user record not found")

    session_data = session_doc.to_dict()
    user_data = user_doc.to_dict()
    turns = session_data.get("turns", [])
    skill_matrix = user_data.get("skill_matrix", {})
    current_turn = turns[question_number - 1]

    # Evaluate answer and generate next adaptive question
    eval_result = evaluate_and_generate_next(
        target_role=session_data.get("target_role", "Software Engineer"),
        question_number=question_number,
        previous_question=current_turn["question"],
        user_answer=user_answer,
        skill_domain=current_turn.get("skill_domain", "Core Architecture"),
        current_skill_matrix=skill_matrix
    )

    # 1. Update the local Skill Matrix in Firestore with the new score
    domain = eval_result.get("evaluated_domain", "Core Architecture")
    skill_matrix[domain] = {
        "score": eval_result.get("score", 7),
        "status": "assessed",
        "last_feedback": eval_result.get("feedback")
    }
    user_ref.update({"skill_matrix": skill_matrix})

    # 2. Update session turn records
    current_turn["answer"] = user_answer
    current_turn["score"] = eval_result.get("score")
    current_turn["feedback"] = eval_result.get("feedback")

    # If completed 5 questions, finalize session
    if question_number >= 5:
        session_ref.update({"turns": turns, "status": "completed", "completed_at": firestore.SERVER_TIMESTAMP})
        return {
            "session_id": session_id,
            "status": "completed",
            "feedback_on_previous": eval_result.get("feedback"),
            "updated_skill_matrix": skill_matrix,
            "message": "Diagnostic completed! Your profile skill matrix is now updated."
        }

    # Otherwise append next turn
    next_turn = {
        "question_number": question_number + 1,
        "question": eval_result["next_question"],
        "skill_domain": eval_result.get("next_skill_domain", "System Design"),
        "answer": None,
        "score": None,
        "feedback": None
    }
    turns.append(next_turn)

    session_ref.update({
        "turns": turns,
        "current_question_number": question_number + 1,
        "updated_at": firestore.SERVER_TIMESTAMP
    })

    return {
        "session_id": session_id,
        "question_number": question_number + 1,
        "question": eval_result["next_question"],
        "skill_domain": eval_result.get("next_skill_domain"),
        "feedback_on_previous": eval_result.get("feedback"),
        "updated_skill_matrix": skill_matrix,
        "total_questions": 5
    }