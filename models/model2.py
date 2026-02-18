"""
model2.py — Parking Violation Detection Module
===============================================
SARAL — Smart Automated Recognition of Automobile Licenses

Provides a single, API-ready entry point:

    detect_parking_violation(image_path) -> dict

The function runs the full illegal-parking detection pipeline on the given
image and returns a structured JSON-serialisable dict with violation details.

Dependencies:
    pip install ultralytics easyocr opencv-python-headless numpy torch

Usage:
    from models.model2 import detect_parking_violation
    result = detect_parking_violation("path/to/image.jpg")
    print(result)
"""

from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SARAL.ParkingDetector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "VIOLATION_DISTANCE": 600,       # px — max horizontal distance to flag a violation
    "VEHICLE_CONF"      : 0.35,      # YOLO minimum confidence for vehicles
    "SIGN_CONF"         : 0.45,      # Minimum confidence for no-parking sign
    "VEHICLE_CLASSES"   : {          # COCO class IDs → human-readable labels
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck",
    },
}

# ---------------------------------------------------------------------------
# Lazy model handles (initialised on first call)
# ---------------------------------------------------------------------------
_yolo_model   = None
_ocr_reader   = None
_sign_detector = None


def _get_yolo():
    """Return (and cache) the YOLOv8n model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8n model (auto-downloads ~6 MB on first run)…")
        _yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n ready.")
    return _yolo_model


def _get_ocr():
    """Return (and cache) the EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import torch
        import easyocr
        use_gpu = torch.cuda.is_available()
        logger.info("Loading EasyOCR (GPU=%s) — first run downloads ~100 MB…", use_gpu)
        _ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        logger.info("EasyOCR ready.")
    return _ocr_reader


def _get_sign_detector():
    """Return (and cache) the NoParkingSignDetector instance."""
    global _sign_detector
    if _sign_detector is None:
        _sign_detector = NoParkingSignDetector(min_cf=CONFIG["SIGN_CONF"])
    return _sign_detector


# ===========================================================================
# OCR — Indian Plate Correction
# ===========================================================================

def _fix_indian_plate(raw: str) -> str:
    """
    Correct common OCR errors in Indian license plates.

    Indian plate format:  SS DD LL NNNN
      SS   = 2 LETTERS  (state code, e.g. MH, KL)
      DD   = 2 DIGITS   (district, e.g. 07)
      LL   = 2 LETTERS  (series, e.g. AB)
      NNNN = 4 DIGITS   (number, e.g. 1234)
    """
    text = re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()
    text = re.sub(r"  +", " ", text)
    tokens = text.split()

    def to_letters(s: str) -> str:
        return s.replace("1", "I").replace("0", "O").replace("8", "B").replace("5", "S")

    def to_digits(s: str) -> str:
        return (s.replace("I", "1").replace("O", "0").replace("B", "8")
                 .replace("S", "5").replace("Z", "2").replace("G", "6")
                 .replace("A", "4").replace("T", "7"))

    if len(tokens) == 4:
        ss, dd, ll, nnnn = tokens
        return f"{to_letters(ss)[:2]} {to_digits(dd)[:2]} {to_letters(ll)[:2]} {to_digits(nnnn)[:4]}"

    if len(tokens) == 1 and len(tokens[0]) >= 8:
        t = tokens[0]
        return f"{to_letters(t[0:2])} {to_digits(t[2:4])} {to_letters(t[4:6])} {to_digits(t[6:10])}"

    if len(tokens) == 2:
        a, b = tokens
        if len(a) == 4 and len(b) == 6:
            return f"{to_letters(a[0:2])} {to_digits(a[2:4])} {to_letters(b[0:2])} {to_digits(b[2:6])}"
        if len(a) == 6 and len(b) == 4:
            return f"{to_letters(a[0:2])} {to_digits(a[2:4])} {to_letters(a[4:6])} {to_digits(b[0:4])}"

    if len(tokens) == 3:
        a, b, c = tokens
        if len(a) == 2 and len(b) == 4 and len(c) == 4:
            return f"{to_letters(a)} {to_digits(b[0:2])} {to_letters(b[2:4])} {to_digits(c)}"
        if len(a) == 4 and len(b) == 2 and len(c) == 4:
            return f"{to_letters(a[0:2])} {to_digits(a[2:4])} {to_letters(b)} {to_digits(c)}"

    return text  # fallback


def _clean_plate(raw: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()
    return _fix_indian_plate(cleaned)


def _is_valid_plate(text: str) -> bool:
    return len(re.sub(r"[^A-Z0-9]", "", text)) >= 4


# ===========================================================================
# No-Parking Sign Detector  (HSV red + circularity heuristic)
# ===========================================================================

class NoParkingSignDetector:
    """Detect circular red no-parking signs using HSV colour + shape analysis."""

    def __init__(self, min_r: int = 18, max_r: int = 400,
                 circ_th: float = 0.50, min_cf: float = 0.45):
        self.min_r   = min_r
        self.max_r   = max_r
        self.circ_th = circ_th
        self.min_cf  = min_cf

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1   = cv2.inRange(hsv, np.array([0,   80,  60]), np.array([12,  255, 255]))
        m2   = cv2.inRange(hsv, np.array([155, 80,  60]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Dict[str, Any]] = []

        for cnt in cnts:
            area  = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if area < np.pi * self.min_r ** 2 or perim == 0:
                continue
            circ = 4 * np.pi * area / perim ** 2
            if circ < self.circ_th:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < 35 or r > self.max_r:
                continue

            x1 = max(0, int(cx - r));  y1 = max(0, int(cy - r))
            x2 = min(image.shape[1] - 1, int(cx + r))
            y2 = min(image.shape[0] - 1, int(cy + r))

            roi = mask[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            fill = np.count_nonzero(roi) / roi.size
            conf = min(1.0, circ * 0.5 + fill * 0.5 + 0.08)
            if conf < self.min_cf:
                continue

            # Reject tail-lights: no-parking signs have a white interior
            roi_color = image[y1:y2, x1:x2]
            if roi_color.size == 0:
                continue
            hsv_roi    = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv_roi,
                                     np.array([0, 0, 180]),
                                     np.array([180, 40, 255]))
            if np.count_nonzero(white_mask) / white_mask.size < 0.15:
                continue

            dets.append({
                "bbox"      : [x1, y1, x2, y2],
                "confidence": round(conf, 2),
                "center"    : (int(cx), int(cy)),
                "radius"    : int(r),
            })

        return self._nms(dets)

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union else 0.0

    def _nms(self, dets: List[Dict[str, Any]], thr: float = 0.4) -> List[Dict[str, Any]]:
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for d in dets:
            if all(self._iou(d["bbox"], k["bbox"]) < thr for k in kept):
                kept.append(d)
        return kept


# ===========================================================================
# Detection Helpers
# ===========================================================================

def _bbox_center(bb: List[int]) -> Tuple[int, int]:
    return ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)


def _horizontal_distance(b1: List[int], b2: List[int]) -> float:
    """Return the horizontal pixel distance between two bounding-box centres."""
    x1 = (b1[0] + b1[2]) // 2
    x2 = (b2[0] + b2[2]) // 2
    return abs(x1 - x2)


def _detect_vehicles(image: np.ndarray) -> List[Dict[str, Any]]:
    """Run YOLOv8n on *image* and return vehicle detections."""
    model = _get_yolo()
    res   = model(image, verbose=False)[0]
    out: List[Dict[str, Any]] = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in CONFIG["VEHICLE_CLASSES"]:
            continue
        conf = float(box.conf[0])
        if conf < CONFIG["VEHICLE_CONF"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        out.append({
            "bbox"      : [x1, y1, x2, y2],
            "label"     : CONFIG["VEHICLE_CLASSES"][cls_id],
            "confidence": round(conf, 2),
        })
    return out


def _read_license_plate(image: np.ndarray, vbbox: List[int]) -> Tuple[str, float]:
    """
    Crop the lower portion of a vehicle bounding box and run EasyOCR to read
    the license plate.

    Returns:
        (plate_text, ocr_confidence)  — plate_text is 'UNREADABLE' on failure.
    """
    reader = _get_ocr()
    x1, y1, x2, y2 = vbbox
    mid_y = y1 + int((y2 - y1) * 0.45)   # lower 55% — where the plate lives
    crop  = image[mid_y:y2, x1:x2]
    if crop.size == 0:
        return "UNREADABLE", 0.0

    # Scale up tiny crops for better OCR resolution
    h, w  = crop.shape[:2]
    scale = max(1, 200 // max(h, 1))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Build multiple pre-processing variants
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2   = cv2.adaptiveThreshold(blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp  = cv2.filter2D(crop, -1, kernel)

    variants = [
        crop,
        sharp,
        cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR),
    ]

    best_text, best_conf = "", 0.0
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

    for region in variants:
        try:
            results = reader.readtext(region, allowlist=allowlist,
                                      detail=1, paragraph=False)
        except Exception:
            continue
        for (_, txt, cf) in results:
            fixed = _clean_plate(txt)
            if cf > best_conf and _is_valid_plate(fixed):
                best_conf = cf
                best_text = fixed

    return (best_text, round(best_conf, 2)) if best_text else ("UNREADABLE", 0.0)


def _overlaps_vehicle(sign_box: List[int],
                      vehicles: List[Dict[str, Any]]) -> bool:
    """Return True if *sign_box* overlaps any vehicle bounding box."""
    sx1, sy1, sx2, sy2 = sign_box
    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["bbox"]
        if sx1 < vx2 and sx2 > vx1 and sy1 < vy2 and sy2 > vy1:
            return True
    return False


# ===========================================================================
# Full Pipeline
# ===========================================================================

def _run_pipeline(image: np.ndarray) -> Tuple[
    List[Dict[str, Any]],   # vehicles
    List[Dict[str, Any]],   # signs
    List[Dict[str, Any]],   # violations (internal format)
]:
    """Run the complete parking-violation detection pipeline on *image*."""
    vehicles  = _detect_vehicles(image)
    raw_signs = _get_sign_detector().detect(image)
    signs     = [s for s in raw_signs
                 if not _overlaps_vehicle(s["bbox"], vehicles)]

    violation_pairs: List[Tuple[Dict, Dict, float]] = []
    for v in vehicles:
        for s in signs:
            if s["confidence"] < CONFIG["SIGN_CONF"]:
                continue
            dist         = _horizontal_distance(v["bbox"], s["bbox"])
            sign_center  = _bbox_center(s["bbox"])[0]
            vehicle_left = v["bbox"][0]
            # Flag vehicles parked to the right of the sign within range
            if vehicle_left > sign_center and dist < CONFIG["VIOLATION_DISTANCE"]:
                violation_pairs.append((v, s, dist))

    violations: List[Dict[str, Any]] = []
    for (v, s, dist) in violation_pairs:
        plate, plate_conf = _read_license_plate(image, v["bbox"])
        violations.append({
            "vehicle" : v,
            "sign"    : s,
            "distance": round(dist, 1),
            "plate"   : plate,
            "plate_confidence": plate_conf,
        })

    return vehicles, signs, violations


# ===========================================================================
# Annotation
# ===========================================================================

def annotate_parking_image(
    image: np.ndarray,
    vehicles: List[Dict[str, Any]],
    signs: List[Dict[str, Any]],
    violations: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw detection results on a copy of *image* and return the annotated copy.

    Draws:
    • Red circle outline around each no-parking sign
    • Orange bounding box + label for each detected vehicle
    • Green plate-text badge on each violating vehicle
    • Dashed distance line from sign centre → vehicle centre
    • Distance label at the midpoint of that line
    • Red fine-amount banner at the top of each violating vehicle box
    """
    out = image.copy()
    h, w = out.shape[:2]
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_sm   = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, w // 400)   # scale line thickness with image size

    # ── Colour palette ───────────────────────────────────────────────────────
    RED    = (0,   30,  220)   # BGR
    ORANGE = (0,  140,  255)
    GREEN  = (50, 200,   50)
    WHITE  = (255, 255, 255)
    BLACK  = (0,   0,    0)
    YELLOW = (0,  220,  220)

    # ── Helper: filled label badge ────────────────────────────────────────────
    def _label(img, text, x, y, bg, fg=WHITE, scale=0.55, th=1):
        (tw, tl), bl = cv2.getTextSize(text, font_sm, scale, th)
        pad = 4
        cv2.rectangle(img, (x - pad, y - tl - pad), (x + tw + pad, y + pad), bg, -1)
        cv2.putText(img, text, (x, y), font_sm, scale, fg, th, cv2.LINE_AA)

    # ── Helper: dashed line ───────────────────────────────────────────────────
    def _dashed_line(img, p1, p2, color, thickness=2, dash=12, gap=8):
        x1, y1 = p1;  x2, y2 = p2
        dist = max(1, int(np.hypot(x2 - x1, y2 - y1)))
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        pos = 0
        drawing = True
        while pos < dist:
            seg = dash if drawing else gap
            end = min(pos + seg, dist)
            if drawing:
                pt1 = (int(x1 + dx * pos), int(y1 + dy * pos))
                pt2 = (int(x1 + dx * end), int(y1 + dy * end))
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
            pos += seg
            drawing = not drawing

    # ── 1. All detected vehicles (light grey box — context only) ─────────────
    violating_vehicle_bboxes = {id(v["vehicle"]): True for v in violations}
    for v in vehicles:
        if id(v) not in violating_vehicle_bboxes:
            x1, y1, x2, y2 = v["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 1)

    # ── 2. No-parking signs ───────────────────────────────────────────────────
    for s in signs:
        cx, cy = s["center"]
        r      = s["radius"]
        # Outer red circle
        cv2.circle(out, (cx, cy), r + 3, RED, thickness + 1, cv2.LINE_AA)
        # Inner thin white ring for contrast
        cv2.circle(out, (cx, cy), r,     WHITE, 1, cv2.LINE_AA)
        # Label
        _label(out, f"No Parking  {int(s['confidence']*100)}%",
               cx - r, cy - r - 22, RED, scale=0.5, th=1)

    # ── 3. Violations: vehicle box + plate + distance line + fine ────────────
    for viol in violations:
        v    = viol["vehicle"]
        s    = viol["sign"]
        dist = int(viol["distance"])
        plate = viol["plate"]

        vx1, vy1, vx2, vy2 = v["bbox"]
        sx_c, sy_c         = s["center"]
        vx_c               = (vx1 + vx2) // 2
        vy_c               = (vy1 + vy2) // 2

        # Orange vehicle box (thick)
        cv2.rectangle(out, (vx1, vy1), (vx2, vy2), ORANGE, thickness + 1)

        # Vehicle label (top-left of box)
        v_label = f"{v['label']}  {int(v['confidence']*100)}%"
        _label(out, v_label, vx1, vy1 - 6, ORANGE, scale=0.55, th=1)

        # Plate badge (bottom of box)
        if plate and plate != "UNREADABLE":
            _label(out, plate, vx1, vy2 + 18, GREEN, scale=0.6, th=2)

        # Dashed distance line: sign centre → vehicle centre
        _dashed_line(out, (sx_c, sy_c), (vx_c, vy_c), YELLOW, thickness)

        # Distance label at midpoint
        mx, my = (sx_c + vx_c) // 2, (sy_c + vy_c) // 2
        _label(out, f"{dist} px", mx, my, BLACK, YELLOW, scale=0.5, th=1)

    # ── 4. Summary banner at top of image ────────────────────────────────────
    n_viol = len(violations)
    if n_viol > 0:
        banner_h = 36
        overlay  = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), RED, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        summary = f"{n_viol} ILLEGAL PARKING VIOLATION{'S' if n_viol > 1 else ''}"
        cv2.putText(out, summary, (10, 24),
                    font, 0.65, WHITE, 2, cv2.LINE_AA)

    return out

def detect_parking_violation(
    image_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the parking-violation detection pipeline on a single image.

    Args:
        image_path:  Absolute or relative path to the input image file.
                     Supported formats: JPEG, PNG, BMP, TIFF, WebP.
        output_path: Optional path where the annotated image will be saved.
                     If provided and violations are detected, the annotated
                     image (with bounding boxes, distance lines, fine badges)
                     is written here and ``annotated_path`` is set in the
                     returned dict.

    Returns:
        A JSON-serialisable dict::

            {
                "violations": [
                    {
                        "type"      : "Illegal Parking",
                        "vehicle"   : "Car",
                        "plate"     : "KL 07 AB 1234",
                        "distance"  : 120,          # px between sign and vehicle
                        "fine"      : "₹500",
                        "confidence": 0.87
                    },
                    ...
                ],
                "vehicle_count" : 2,
                "sign_detected" : true,
                "annotated_path": "/abs/path/to/annotated.jpg"  # if output_path given
            }

        On error the function returns::

            {
                "violations"   : [],
                "vehicle_count": 0,
                "sign_detected": false,
                "error"        : "<error message>"
            }

    Raises:
        Does NOT raise — all exceptions are caught and returned in the
        ``"error"`` field so the caller can treat this as a pure function.
    """
    # --- Validate input ---------------------------------------------------
    if not os.path.isfile(image_path):
        logger.error("Image not found: %s", image_path)
        return {
            "violations"   : [],
            "vehicle_count": 0,
            "sign_detected": False,
            "error"        : f"File not found: {image_path}",
        }

    image = cv2.imread(image_path)
    if image is None:
        logger.error("Could not decode image: %s", image_path)
        return {
            "violations"   : [],
            "vehicle_count": 0,
            "sign_detected": False,
            "error"        : f"Could not decode image: {image_path}",
        }

    logger.info("Processing image: %s  (%dx%d px)",
                os.path.basename(image_path), image.shape[1], image.shape[0])

    # --- Run pipeline -----------------------------------------------------
    try:
        vehicles, signs, raw_violations = _run_pipeline(image)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return {
            "violations"   : [],
            "vehicle_count": 0,
            "sign_detected": False,
            "error"        : str(exc),
        }

    # --- Build structured output -----------------------------------------
    violations_out: List[Dict[str, Any]] = []
    for viol in raw_violations:
        v    = viol["vehicle"]
        conf = round(
            (v["confidence"] + viol["sign"]["confidence"]) / 2, 2
        )
        violations_out.append({
            "type"      : "Illegal Parking",
            "vehicle"   : v["label"],
            "plate"     : viol["plate"],
            "distance"  : int(viol["distance"]),
            "confidence": conf,
        })

    result: Dict[str, Any] = {
        "violations"   : violations_out,
        "vehicle_count": len(vehicles),
        "sign_detected": len(signs) > 0,
    }

    # --- Annotate and save image ------------------------------------------
    if output_path:
        try:
            annotated = annotate_parking_image(image, vehicles, signs, raw_violations)
            cv2.imwrite(output_path, annotated)
            result["annotated_path"] = output_path
            logger.info("Annotated image saved: %s", output_path)
        except Exception as ae:
            logger.warning("Failed to save annotated image: %s", ae)

    logger.info(
        "Done — vehicles: %d | signs: %d | violations: %d",
        len(vehicles), len(signs), len(violations_out),
    )
    return result


# ===========================================================================
# CLI convenience
# ===========================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python model2.py <image_path>")
        sys.exit(1)

    output = detect_parking_violation(sys.argv[1])
    print(json.dumps(output, indent=2, ensure_ascii=False))