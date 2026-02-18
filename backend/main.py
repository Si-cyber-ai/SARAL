"""
SARAL — FastAPI Backend
Serves the AI detection pipeline, report management, auth, and karma system.
"""

import os
import sys
import uuid
import shutil
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root and models dir to path so we can import model.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)

import database as db

# ─── Initialise the database (create tables + seed users) ───
db.init_db()

# ─── Try importing the AI model (graceful fallback if deps missing) ───
try:
    from models.model import process_image, process_video, get_input_type, save_annotated_image
    AI_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] AI model not available: {e}")
    print("[WARNING] The API will work but /api/analyze will return mock results.")
    AI_MODEL_AVAILABLE = False

# ─── App Setup ───
app = FastAPI(title="SARAL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded files as static
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Also serve the frontend
FRONTEND_DIR = PROJECT_ROOT
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")

KARMA_POINTS_PER_APPROVAL = 150


# ===========================================================================
# AUTH ROUTES
# ===========================================================================

@app.post("/api/auth/signin")
async def signin(email: str = Form(...), password: str = Form(...), role: str = Form(...)):
    user = db.get_user_by_email(email)
    if not user or user["password"] != password or user["role"] != role:
        raise HTTPException(status_code=401, detail="Invalid credentials or role mismatch")
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "karma_points": user["karma_points"],
            "city": user.get("city", ""),
        },
    }


@app.post("/api/auth/signup")
async def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
):
    existing = db.get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = db.create_user(name, email, password, role)
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "karma_points": user["karma_points"],
        },
    }


@app.get("/api/auth/user/{user_id}")
async def get_user(user_id: int):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "karma_points": user["karma_points"],
        "city": user.get("city", ""),
    }


@app.put("/api/auth/user/{user_id}")
async def update_profile(
    user_id: int,
    name: str = Form(...),
    email: str = Form(...),
    city: str = Form(""),
    password: str = Form(None),
):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.update_user_profile(user_id, name, email, city, password)
    updated = db.get_user_by_id(user_id)
    return {"success": True, "user": {
        "id": updated["id"], "name": updated["name"],
        "email": updated["email"], "city": updated.get("city", ""),
        "karma_points": updated["karma_points"],
    }}


# ===========================================================================
# ANALYZE (AI PIPELINE) ROUTE
# ===========================================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    violation_type: str = Form(""),
    location: str = Form(""),
    description: str = Form(""),
    manual_plate: str = Form(""),
):
    """
    Accept image/video upload, run AI model, store report, return result.
    """
    # Validate user exists
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    media_url = f"/uploads/{unique_name}"

    # Run AI model
    plate_number = ""
    confidence = 0.0
    detected_violation = violation_type
    helmet_info = ""
    annotated_url = ""
    ai_results = None  # keep raw results for annotated image generation

    if AI_MODEL_AVAILABLE:
        try:
            input_type = get_input_type(file_path)

            if input_type == "image":
                results = process_image(file_path, show_visual=False)
            else:
                result = process_video(file_path, num_frames=10)
                results = [result] if result else []

            if results and len(results) > 0:
                ai_results = results
                best = results[0]
                plate_number = best.get("plate_text", "")
                confidence = best.get("avg_confidence", 0.0)

                # Check for helmet detections
                helmet_dets = best.get("helmet_detections", [])
                no_helmet = [h for h in helmet_dets if "no" in h.get("class", "").lower() or "without" in h.get("class", "").lower()]
                with_helmet = [h for h in helmet_dets if h not in no_helmet]

                if no_helmet and not detected_violation:
                    detected_violation = "No Helmet"
                elif not detected_violation and plate_number:
                    detected_violation = "Traffic Violation"

                if helmet_dets:
                    helmet_info = f"{len(with_helmet)} with helmet, {len(no_helmet)} without"

                # Use YOLO confidence if OCR confidence is 0
                if confidence == 0.0 and best.get("yolo_confidence", 0) > 0:
                    confidence = best["yolo_confidence"]

                # If plate confidence is still 0 but helmet was detected, use helmet confidence
                if confidence == 0.0 and helmet_dets:
                    best_helmet_conf = max(h.get("confidence", 0.0) for h in helmet_dets)
                    if best_helmet_conf > 0:
                        confidence = best_helmet_conf

                # Generate annotated image with bounding boxes
                try:
                    image_rgb = best.get("image_rgb")
                    if image_rgb is not None:
                        annotated_name = f"annotated_{uuid.uuid4().hex}.png"
                        annotated_path = os.path.join(UPLOAD_DIR, annotated_name)
                        save_annotated_image(image_rgb, results, annotated_path)
                        annotated_url = f"/uploads/{annotated_name}"
                except Exception as ae:
                    print(f"[WARNING] Failed to save annotated image: {ae}")

        except Exception as e:
            print(f"[ERROR] AI model failed: {e}")
            # Continue with empty results — report still gets created
    else:
        # Mock result when model is not available
        plate_number = "KL 11 AB 1234"
        confidence = 0.91
        if not detected_violation:
            detected_violation = "Traffic Violation"

    if not detected_violation:
        detected_violation = "Unknown Violation"

    # Normalize confidence to percentage (0-100)
    confidence_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)

    # ─── Plate match verification ───
    import re as _re
    def _normalize_plate(s: str) -> str:
        return _re.sub(r'[^A-Z0-9]', '', s.upper())

    norm_ocr = _normalize_plate(plate_number)
    norm_manual = _normalize_plate(manual_plate) if manual_plate else ""
    plate_match = None
    if norm_manual and norm_ocr:
        plate_match = norm_manual == norm_ocr

    # Auto-reject if AI confidence is below 30%
    auto_status = "Auto-Rejected" if confidence_pct < 30 else "Under Review"

    # Auto-reject "No Helmet" reports when AI found everyone wearing helmets
    # or when AI model is unavailable (can't verify the claim)
    _vt = (violation_type or "").lower().replace(" ", "")
    _is_helmet_violation = _vt in ("nohelmet", "helmet")
    if _is_helmet_violation:
        if not AI_MODEL_AVAILABLE:
            # No AI to verify — reject since we can't confirm helmet violation
            auto_status = "Auto-Rejected"
        elif helmet_info:
            try:
                # helmet_info format: "X with helmet, Y without"
                parts = helmet_info.split(",")
                without_count = int(parts[1].strip().split()[0]) if len(parts) >= 2 else -1
                if without_count == 0:
                    auto_status = "Auto-Rejected"
            except (ValueError, IndexError):
                pass

    # Store report in database
    report = db.create_report(
        user_id=user_id,
        plate_number=plate_number,
        violation_type=detected_violation,
        confidence=confidence_pct,
        media_url=media_url,
        annotated_url=annotated_url,
        location=location,
        description=description,
        helmet_detected=helmet_info,
        status=auto_status,
        manual_plate=norm_manual,
    )

    return {
        "report_id": report["id"],
        "violation": report["violation_type"],
        "plate": report["plate_number"],
        "confidence": report["confidence"],
        "status": report["status"],
        "auto_rejected": auto_status == "Auto-Rejected",
        "media_url": report["media_url"],
        "annotated_url": report.get("annotated_url", ""),
        "helmet_detected": helmet_info,
        "created_at": report["created_at"],
        "manual_plate": norm_manual if norm_manual else None,
        "ocr_plate": norm_ocr if norm_ocr else None,
        "match": plate_match,
    }


# ===========================================================================
# REPORT ROUTES
# ===========================================================================

@app.get("/api/reports/user/{user_id}")
async def get_user_reports(user_id: int):
    """Get all reports for a specific user."""
    reports = db.get_reports_by_user(user_id)
    return {"reports": reports}


@app.get("/api/reports/stats/{user_id}")
async def get_user_report_stats(user_id: int):
    """Get report statistics for a user."""
    stats = db.get_report_stats(user_id)
    user = db.get_user_by_id(user_id)
    return {
        **stats,
        "karma_points": user["karma_points"] if user else 0,
    }


@app.get("/api/reports/pending")
async def get_pending():
    """Get all reports with status 'Under Review' (for authority)."""
    reports = db.get_pending_reports()
    return {"reports": reports}


@app.get("/api/reports/all")
async def get_all():
    """Get all reports (for authority archive)."""
    reports = db.get_all_reports()
    return {"reports": reports}


@app.get("/api/reports/{report_id}")
async def get_report(report_id: int):
    report = db.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.patch("/api/reports/{report_id}/plate")
async def update_plate(report_id: int, manual_plate: str = Form(...)):
    """Update the citizen-reported plate number on an existing report."""
    import re as _re
    def _normalize(s: str) -> str:
        return _re.sub(r'[^A-Z0-9]', '', s.upper())
    report = db.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    norm = _normalize(manual_plate)
    updated = db.update_report_manual_plate(report_id, norm)
    return {"success": True, "manual_plate": norm, "report": updated}


@app.patch("/api/reports/{report_id}/details")
async def update_report_details(
    report_id: int,
    location: str = Form(None),
    description: str = Form(None),
    violation_type: str = Form(None),
    manual_plate: str = Form(None),
):
    """Update report details (location, description, violation type, plate) at submit time."""
    report = db.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    import re as _re
    norm_plate = None
    if manual_plate is not None:
        norm_plate = _re.sub(r'[^A-Z0-9]', '', manual_plate.upper())

    updated = db.update_report_details(
        report_id,
        location=location,
        description=description,
        violation_type=violation_type,
        manual_plate=norm_plate,
    )

    # Re-evaluate auto-reject for "No Helmet" violations after details update
    refreshed = db.get_report_by_id(report_id)
    if refreshed and refreshed["status"] == "Under Review":
        _vt = (refreshed.get("violation_type") or "").lower().replace(" ", "")
        _is_helmet = _vt in ("nohelmet", "helmet")
        helmet_info = refreshed.get("helmet_detected") or ""
        if _is_helmet and helmet_info:
            try:
                parts = helmet_info.split(",")
                without_count = int(parts[1].strip().split()[0]) if len(parts) >= 2 else -1
                if without_count == 0:
                    db.update_report_status(report_id, "Auto-Rejected")
                    refreshed = db.get_report_by_id(report_id)
            except (ValueError, IndexError):
                pass

    return {"success": True, "report": refreshed or updated}


# ===========================================================================
# AUTHORITY ACTION ROUTES
# ===========================================================================

@app.post("/api/reports/{report_id}/approve")
async def approve_report(report_id: int):
    """Approve a report: status → Approved, +150 karma to reporter."""
    report = db.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report["status"] != "Under Review":
        raise HTTPException(status_code=400, detail=f"Report already {report['status']}")

    # Update status
    updated = db.update_report_status(report_id, "Approved")

    # Award karma points to the reporter
    new_balance = db.update_user_karma(report["user_id"], KARMA_POINTS_PER_APPROVAL)

    return {
        "success": True,
        "report": updated,
        "karma_awarded": KARMA_POINTS_PER_APPROVAL,
        "user_karma_balance": new_balance,
    }


@app.post("/api/reports/{report_id}/reject")
async def reject_report(report_id: int):
    """Reject a report: status → Rejected."""
    report = db.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report["status"] != "Under Review":
        raise HTTPException(status_code=400, detail=f"Report already {report['status']}")

    updated = db.update_report_status(report_id, "Rejected")

    return {
        "success": True,
        "report": updated,
    }


# ===========================================================================
# REWARDS ROUTES
# ===========================================================================

REWARDS_CATALOGUE = [
    {"id": "RW-001", "title": "FASTag Recharge", "desc": "Get ₹200 FASTag credit for highway tolls", "cost": 150, "icon": "fastag"},
    {"id": "RW-002", "title": "Metro Pass", "desc": "5-day unlimited metro rides in your city", "cost": 120, "icon": "metro"},
    {"id": "RW-003", "title": "Fuel Voucher", "desc": "₹300 fuel discount at partner stations", "cost": 200, "icon": "fuel"},
    {"id": "RW-004", "title": "Gift Card", "desc": "₹500 Amazon/Flipkart gift card", "cost": 300, "icon": "gift"},
]


@app.get("/api/rewards/catalogue")
async def rewards_catalogue():
    return {"catalogue": REWARDS_CATALOGUE}


@app.post("/api/rewards/redeem")
async def redeem_reward(
    user_id: int = Form(...),
    reward_id: str = Form(...),
):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    reward = next((r for r in REWARDS_CATALOGUE if r["id"] == reward_id), None)
    if not reward:
        raise HTTPException(status_code=404, detail="Reward not found")

    if user["karma_points"] < reward["cost"]:
        raise HTTPException(status_code=400, detail="Insufficient karma points")

    # Deduct points and record redemption
    new_balance = db.update_user_karma(user_id, -reward["cost"])
    redeemed = db.create_redeemed_reward(user_id, reward_id, reward["title"], reward["cost"])

    return {
        "success": True,
        "new_balance": new_balance,
        "redeemed": redeemed,
    }


@app.get("/api/rewards/history/{user_id}")
async def rewards_history(user_id: int):
    history = db.get_redeemed_rewards(user_id)
    return {"history": history}


# ===========================================================================
# AUTHORITY STATS
# ===========================================================================

@app.get("/api/authority/stats")
async def authority_stats():
    stats = db.get_authority_stats()
    return stats


# ===========================================================================
# HEALTH CHECK
# ===========================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ai_model_available": AI_MODEL_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
