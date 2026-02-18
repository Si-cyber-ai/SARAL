# -*- coding: utf-8 -*-
"""
SARAL — Smart Automated Recognition of Automobile Licenses

A complete pipeline for detecting and reading Indian license plates from
images and videos.

Pipeline overview:
    1. YOLOv8 (hosted on Roboflow) detects license-plate bounding boxes.
    2. Detected regions are cropped with configurable padding.
    3. Each crop is preprocessed into multiple image variants (CLAHE, adaptive
       threshold, morphological, inverted, sharpened, bilateral).
    4. EasyOCR reads text from every variant; the best result is selected by a
       scoring function that rewards valid Indian plate format and high OCR
       confidence.
    5. OCR errors are corrected using position-aware character-confusion maps
       and exhaustive district/series/number split scoring.
    6. For video inputs, N frames are sampled evenly, each frame is processed
       independently, and majority voting across frames picks the final plate
       text.

Usage (CLI):
    python saral_pretrained_colab.py path/to/image_or_video.jpg
    python saral_pretrained_colab.py path/to/video.mp4 --num-frames 20

Dependencies:
    pip install inference-sdk easyocr opencv-python matplotlib numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from inference_sdk import InferenceHTTPClient
from matplotlib import patches

# ===========================================================================
#  Logging Configuration
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SARAL")

# ===========================================================================
#  Global Configuration
# ===========================================================================

# Roboflow model identifier for Indian license-plate detection
ROBOFLOW_MODEL_ID: str = "indian-plate/1"

# Roboflow model identifier for helmet detection
HELMET_MODEL_ID: str = "helmet-detection-tiuol/1"

# API key — hardcoded for Google Colab usage
ROBOFLOW_API_KEY: str = "0rI7Tij9roXu1Ik61Z4c"

# Accepted file extensions
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Characters accepted by EasyOCR for plate recognition
PLATE_ALLOWLIST: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

# Minimum per-segment OCR confidence before a segment is considered noise
MIN_SEGMENT_CONFIDENCE: float = 0.25

# Default number of evenly-spaced frames to sample from a video
DEFAULT_NUM_FRAMES: int = 15

# Majority-voting bonus added for each extra frame that agrees on a plate text
VOTE_FREQUENCY_BONUS: float = 1.5

# ===========================================================================
#  Roboflow & EasyOCR Clients  (initialised lazily — see _get_*())
# ===========================================================================
_roboflow_client: Optional[InferenceHTTPClient] = None
_easyocr_reader: Optional[easyocr.Reader] = None


def _get_roboflow_client() -> InferenceHTTPClient:
    """Return (and cache) the Roboflow inference client."""
    global _roboflow_client
    if _roboflow_client is None:
        logger.info("Initialising Roboflow client (API URL: serverless.roboflow.com)")
        _roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
    return _roboflow_client


def _get_easyocr_reader() -> easyocr.Reader:
    """Return (and cache) the EasyOCR reader for English text."""
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initialising EasyOCR reader (lang='en') — first call may download models")
        _easyocr_reader = easyocr.Reader(["en"])
    return _easyocr_reader


# ===========================================================================
#  Indian License-Plate Format Constants
# ===========================================================================
# Standard format:  SS DD A(A)(A) NNNN
#   SS   = state code  (2 uppercase letters, e.g. MH, DL, KA)
#   DD   = district     (1–2 digits)
#   A–AAA = series      (1–3 uppercase letters)
#   NNNN = number       (1–4 digits)
# Examples: MH 46 X 9996,  KA 01 AB 1234,  DL 9 SL 8346

INDIAN_PLATE_REGEX: re.Pattern = re.compile(
    r"([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})"
)

# All valid Indian state / Union Territory RTO codes
INDIAN_STATE_CODES: set[str] = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# ---------------------------------------------------------------------------
#  OCR Character-Confusion Maps
#
#  When a character appears in a position where a digit is expected, these
#  maps list the most likely digit it was confused with (and vice-versa).
# ---------------------------------------------------------------------------
DIGIT_CANDIDATES: Dict[str, List[str]] = {
    "O": ["0"], "Q": ["0"], "D": ["0"],
    "I": ["1"], "L": ["4", "1"], "l": ["1"],
    "Z": ["2"], "z": ["2"],
    "E": ["3"],
    "A": ["4"], "H": ["4"],
    "S": ["5"], "s": ["5"],
    "G": ["6"], "b": ["6"],
    "T": ["7"],
    "B": ["8"],
    "g": ["9"], "q": ["9"],
}

LETTER_CANDIDATES: Dict[str, List[str]] = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L"],
    "2": ["Z"],
    "3": ["E"],
    "4": ["A", "H"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
}

# Single-character visual-similarity groups (letters that look alike)
SIMILAR_LETTERS: Dict[str, List[str]] = {
    "H": ["M", "N", "W"], "M": ["H", "N", "W"],
    "N": ["H", "M"],      "W": ["M", "H"],
    "C": ["G", "O"],      "G": ["C", "O"],
    "P": ["R", "B"],      "R": ["P", "B"],
    "B": ["R", "P", "D"], "D": ["O", "B"],
    "U": ["V"],           "V": ["U"],
    "T": ["A", "I"],      "A": ["H"],
    "Y": ["V"],           "I": ["T", "L"],
    "E": ["F"],           "F": ["E", "P"],
}


# ===========================================================================
#  Helper: Input-Type Detection
# ===========================================================================

def get_input_type(file_path: str) -> str:
    """
    Classify *file_path* as ``'image'`` or ``'video'`` based on its extension.

    Raises:
        ValueError: If the extension is not in IMAGE_EXTENSIONS or VIDEO_EXTENSIONS.
    """
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported file format: '{ext}'")


# ===========================================================================
#  Stage 0 — Helmet Detection
# ===========================================================================

def detect_helmets(image_path: str) -> dict:
    """
    Send *image_path* to the Roboflow-hosted helmet-detection model.

    Returns:
        Raw API response dict containing ``'predictions'`` (list of bounding
        boxes with ``x, y, width, height, confidence, class``).
        Classes typically include 'helmet' and 'no-helmet' (or similar).
    """
    logger.info("Helmet detection inference on: %s", image_path)
    client = _get_roboflow_client()
    result = client.infer(image_path, model_id=HELMET_MODEL_ID)
    preds = result.get("predictions", [])
    logger.info("  -> Found %d helmet-related detection(s)", len(preds))
    return result


# ===========================================================================
#  Stage 1 — YOLOv8 Plate Detection
# ===========================================================================

def detect_plates_yolov8(image_path: str) -> dict:
    """
    Send *image_path* to the Roboflow-hosted YOLOv8 model for license-plate
    detection.

    Returns:
        Raw API response dict containing ``'predictions'`` (list of bounding
        boxes with ``x, y, width, height, confidence, class``).
    """
    logger.info("YOLOv8 inference on: %s", image_path)
    client = _get_roboflow_client()
    result = client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)
    n_plates = len(result.get("predictions", []))
    logger.info("  -> Found %d plate(s)", n_plates)
    return result


# ===========================================================================
#  Stage 2 — Crop Detected Plate Regions
# ===========================================================================

def crop_plates_from_predictions(
    image_rgb: np.ndarray,
    predictions: List[dict],
    padding_pct: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Extract sub-images for each detected plate from *image_rgb*.

    Each prediction uses *center-x, center-y, width, height* format.  An
    extra ``padding_pct`` fraction of the width/height is added around each
    crop to avoid cutting off plate edges.

    Returns:
        List of dicts, each containing:
        ``crop`` (np.ndarray), ``bbox`` (x_min, y_min, x_max, y_max),
        ``confidence`` (float), ``class`` (str).
    """
    crops: List[Dict[str, Any]] = []
    img_h, img_w = image_rgb.shape[:2]

    for pred in predictions:
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        confidence = pred["confidence"]

        # Add padding around the detected region
        pad_w = int(w * padding_pct)
        pad_h = int(h * padding_pct)

        # Convert centre → corner coordinates with padding, clamped to image bounds
        x_min = max(int(cx - w / 2) - pad_w, 0)
        y_min = max(int(cy - h / 2) - pad_h, 0)
        x_max = min(int(cx + w / 2) + pad_w, img_w)
        y_max = min(int(cy + h / 2) + pad_h, img_h)

        crop_img = image_rgb[y_min:y_max, x_min:x_max]
        crops.append({
            "crop": crop_img,
            "bbox": (x_min, y_min, x_max, y_max),
            "confidence": confidence,
            "class": pred.get("class", "plate"),
        })

    return crops


# ===========================================================================
#  Stage 3a — Image Pre-processing (multiple variants for OCR robustness)
# ===========================================================================

def preprocess_plate_image(
    crop_rgb: np.ndarray,
    target_height: int = 200,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate six preprocessed greyscale variants of a plate crop.

    Small crops are upscaled to *target_height* first so that OCR has enough
    pixel data to work with.

    Returns:
        List of ``(variant_label, processed_greyscale_image)`` tuples.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # Up-scale tiny crops for better OCR resolution
    h, w = gray.shape
    if h < target_height:
        scale = target_height / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    variants: List[Tuple[str, np.ndarray]] = []

    # 1. CLAHE + Otsu thresholding
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    _, otsu = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", otsu))

    # 2. De-noise + adaptive threshold
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl2 = clahe2.apply(denoised)
    adapt = cv2.adaptiveThreshold(
        cl2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    variants.append(("adaptive", adapt))

    # 3. Morphological close → open cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    variants.append(("morph", morph))

    # 4. Inverted (for white-on-black plates)
    variants.append(("inverted", cv2.bitwise_not(otsu)))

    # 5. Sharpened high-contrast greyscale (no binarisation)
    sharp_kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    sharpened = cv2.filter2D(cl, -1, sharp_kernel)
    variants.append(("sharpened", sharpened))

    # 6. Bilateral filter + CLAHE (edge-preserving smoothing)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    clahe3 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl3 = clahe3.apply(bilateral)
    variants.append(("bilateral", cl3))

    # 7. High-contrast CLAHE + aggressive Otsu (helps with faded plates)
    clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    cl4 = clahe4.apply(gray)
    _, otsu2 = cv2.threshold(cl4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_strong", otsu2))

    # 8. Gaussian blur + adaptive threshold (reduces speckle noise)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adapt2 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8,
    )
    variants.append(("gauss_adaptive", adapt2))

    return variants


# ===========================================================================
#  Stage 3b — OCR Error-Correction Helpers
# ===========================================================================

def _char_letter_candidates(c: str) -> List[str]:
    """
    Return all plausible *letter* interpretations of character *c*, ordered
    from most likely to least likely.  Handles digit→letter and similar-letter
    confusions.
    """
    c = c.upper()
    cands: List[str] = []
    if c.isalpha():
        cands.append(c)
        cands.extend(SIMILAR_LETTERS.get(c, []))
    if c in LETTER_CANDIDATES:
        cands.extend(LETTER_CANDIDATES[c])
    if not cands:
        cands.append(c)
    # De-duplicate while preserving order
    seen: set[str] = set()
    return [x for x in cands if not (x in seen or seen.add(x))]


def find_closest_state_code(two_chars: str) -> str:
    """
    Given two characters, find the closest valid Indian RTO state code.

    Tries all one-character edits using SIMILAR_LETTERS and LETTER_CANDIDATES.
    Returns the original string unchanged if no close match is found.
    """
    two_chars = two_chars.upper()
    if two_chars in INDIAN_STATE_CODES:
        return two_chars

    c0, c1 = two_chars[0], two_chars[1]
    alts_0 = [c0] + SIMILAR_LETTERS.get(c0, []) + LETTER_CANDIDATES.get(c0, [])
    alts_1 = [c1] + SIMILAR_LETTERS.get(c1, []) + LETTER_CANDIDATES.get(c1, [])

    candidates: List[Tuple[int, str]] = []
    for a0 in alts_0:
        for a1 in alts_1:
            code = a0 + a1
            if code in INDIAN_STATE_CODES:
                dist = (0 if a0 == c0 else 1) + (0 if a1 == c1 else 1)
                candidates.append((dist, code))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    return two_chars


def _resolve_state_code(state_raw: str) -> str:
    """
    Find the best valid Indian state code for two raw OCR characters.

    Exhaustively tries all plausible letter interpretations (including
    digit↔letter substitutions like ``0→O/D``, ``1→I``, ``8→B``) and picks
    the candidate with the fewest character changes.
    """
    c0, c1 = state_raw[0].upper(), state_raw[1].upper()
    alts_0 = _char_letter_candidates(c0)
    alts_1 = _char_letter_candidates(c1)

    best_code: Optional[str] = None
    best_dist = 999
    for a0 in alts_0:
        for a1 in alts_1:
            code = a0 + a1
            if code in INDIAN_STATE_CODES:
                dist = (0 if a0 == c0 else 1) + (0 if a1 == c1 else 1)
                if dist < best_dist:
                    best_dist = dist
                    best_code = code

    if best_code:
        return best_code

    # Fallback: convert digits to their first letter candidate
    fallback = "".join(
        (LETTER_CANDIDATES[c][0] if c in LETTER_CANDIDATES else c)
        for c in [c0, c1]
    )
    return find_closest_state_code(fallback)


def correct_plate_text(raw_text: str) -> str:
    """
    Apply Indian license-plate format rules and OCR error-correction to
    *raw_text*.

    Strategy:
        1. Normalise (uppercase, strip symbols/whitespace).
        2. Remove ``"IND"`` country marker if present.
        3. Parse the first two characters as a state code.
        4. Try all combinations of district length (1 or 2 digits) × series
           length (1–3 letters) × remaining number digits.  Score each split
           by how many characters already match their expected type (digit vs
           letter) and by adherence to common plate conventions (4-digit
           number, 2-digit district, etc.).
        5. Return the highest-scoring candidate formatted as
           ``"SS DD AAA NNNN"``.
    """
    # --- Normalise --------------------------------------------------------
    text = raw_text.upper().strip()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip "IND" country marker that OCR often reads from plate emblems
    text = re.sub(r"\bIND\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Collapse to a single string for positional parsing
    raw = re.sub(r"\s+", "", text)

    # Also strip leading "IND" if stuck to the plate text without space
    if raw.startswith("IND") and len(raw) > 7:
        raw = raw[3:]

    if len(raw) < 5:
        return text  # Too short to be a valid plate

    # --- State code (positions 0–1) ---------------------------------------
    state_raw = raw[:2]
    rest = raw[2:]
    state_final = _resolve_state_code(state_raw)

    # --- Exhaustive district / series / number split ----------------------
    best_candidate: Optional[str] = None
    best_score: float = -1.0

    for district_len in (1, 2):
        if len(rest) < district_len + 2:
            continue  # need ≥1 series char + ≥1 number char after district

        district_part = rest[:district_len]
        after_district = rest[district_len:]

        # Convert district characters → digits
        d_chars: List[str] = []
        valid_district = True
        for c in district_part:
            if c.isdigit():
                d_chars.append(c)
            elif c in DIGIT_CANDIDATES:
                d_chars.append(DIGIT_CANDIDATES[c][0])
            else:
                valid_district = False
                break
        if not valid_district:
            continue
        district_str = "".join(d_chars)

        # Try every plausible series / number split point
        max_series_len = min(3, max(1, len(after_district) - 1))
        for series_len in range(1, max_series_len + 1):
            series_part = after_district[:series_len]
            number_part = after_district[series_len:]
            if not number_part:
                continue

            # Build all plausible letter interpretations for each series char
            series_char_options: List[List[str]] = []
            valid_series = True
            for c in series_part:
                opts = _char_letter_candidates(c)
                if not opts:
                    valid_series = False
                    break
                series_char_options.append(opts)
            if not valid_series:
                continue

            # Convert number characters → digits
            n_chars: List[str] = []
            valid_number = True
            for c in number_part:
                if c.isdigit():
                    n_chars.append(c)
                elif c in DIGIT_CANDIDATES:
                    n_chars.append(DIGIT_CANDIDATES[c][0])
                else:
                    valid_number = False
                    break
            if not valid_number or not n_chars:
                continue

            number_str = "".join(n_chars)

            # Try all plausible series letter combinations
            for series_combo in product(*series_char_options):
                series_str = "".join(series_combo)

                # --- Score this parse ---
                score: float = 0.0
                # Reward characters that already match their expected type
                score += sum(2 for c in district_part if c.isdigit())
                score += sum(2 for c in series_part  if c.isalpha())
                score += sum(2 for c in number_part  if c.isdigit())
                # Most plates have 4-digit number
                if len(number_str) == 4:
                    score += 5
                elif len(number_str) == 3:
                    score += 1
                # Most series codes are 1–2 letters; penalise 3-letter series
                if len(series_str) <= 2:
                    score += 3
                else:
                    score -= 2
                # Strong preference for 2-digit district (nearly all Indian plates)
                if len(district_str) == 2:
                    score += 4
                elif len(district_str) == 1:
                    score -= 1
                # Slight preference for early-alphabet series (more common in Indian plates)
                score += sum(0.15 for c in series_str if c in "ABCDEFGHIJKLM")

                candidate = f"{state_final} {district_str} {series_str} {number_str}"

                # Large bonus if the candidate matches the standard plate regex
                if INDIAN_PLATE_REGEX.search(candidate.replace(" ", "")):
                    score += 5

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

    # --- Final formatting -------------------------------------------------
    if best_candidate:
        match = INDIAN_PLATE_REGEX.search(best_candidate.replace(" ", ""))
        if match:
            state, dist, series, num = match.groups()
            return f"{state} {dist} {series} {num}"
        return best_candidate

    # Fallback: return state code + remaining characters as-is
    return f"{state_final} {rest}".strip()


# ===========================================================================
#  Stage 3c — EasyOCR Text Reading
# ===========================================================================

def read_text_from_crops(
    crops: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run EasyOCR on multiple preprocessed variants of each plate crop and
    select the variant whose output best matches the Indian plate format.

    Returns:
        One result dict per crop containing ``texts``, ``confidences``,
        ``boxes``, ``full_text`` (corrected), ``raw_text``, and ``variant``.
    """
    reader = _get_easyocr_reader()
    ocr_results: List[Dict[str, Any]] = []

    for crop_info in crops:
        crop_img = crop_info["crop"]
        variants = preprocess_plate_image(crop_img)

        best_score: float = -1.0
        best_result: Dict[str, Any] = {
            "texts": [], "confidences": [], "boxes": [], "full_text": "",
        }

        for label, processed in variants:
            results = reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist=PLATE_ALLOWLIST,
            )

            texts: List[str] = []
            confidences: List[float] = []
            boxes: List[np.ndarray] = []

            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
                boxes.append(np.array(bbox))

            # Evaluate two segment sets: all segments, and high-confidence only
            segment_sets = [(texts, confidences, boxes)]
            if len(texts) > 1:
                filtered = [
                    (t, c, b)
                    for t, c, b in zip(texts, confidences, boxes)
                    if c >= MIN_SEGMENT_CONFIDENCE
                ]
                if filtered and len(filtered) < len(texts):
                    f_texts, f_confs, f_boxes = zip(*filtered)
                    segment_sets.append(
                        (list(f_texts), list(f_confs), list(f_boxes))
                    )

            for seg_texts, seg_confs, seg_boxes in segment_sets:
                raw_text = " ".join(seg_texts) if seg_texts else ""
                corrected = correct_plate_text(raw_text)
                avg_conf = float(np.mean(seg_confs)) if seg_confs else 0.0

                # Score = OCR confidence + format-validity bonus − noise penalty
                score = avg_conf
                if INDIAN_PLATE_REGEX.search(corrected.replace(" ", "")):
                    score += 2.0
                if seg_confs and min(seg_confs) < 0.20:
                    score -= 0.5

                if score > best_score:
                    best_score = score
                    best_result = {
                        "texts": seg_texts,
                        "confidences": seg_confs,
                        "boxes": seg_boxes,
                        "full_text": corrected,
                        "raw_text": raw_text,
                        "variant": label,
                    }

        ocr_results.append(best_result)

    return ocr_results


# ===========================================================================
#  Full Pipeline: Single Image
# ===========================================================================

def process_image(
    image_input: str | np.ndarray,
    show_visual: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the complete detection → crop → OCR pipeline on a single image.

    Args:
        image_input: Either a file-system path (str) or an RGB ``np.ndarray``
                     frame (e.g. from a video).
        show_visual: If ``True``, display matplotlib plots of detections and
                     cropped plates.

    Returns:
        List of result dicts, one per detected plate, each containing:
        ``plate_text``, ``raw_text``, ``avg_confidence``, ``yolo_confidence``,
        ``valid_format``, ``variant``, ``texts``, ``confidences``,
        ``crop_info``, ``image_rgb``.
    """
    # --- Load / prepare the image ----------------------------------------
    temp_path: Optional[str] = None

    if isinstance(image_input, np.ndarray):
        image_rgb = image_input
        # The Roboflow API requires a file path; write frame to a temp file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg", prefix="saral_frame_")
        os.close(fd)
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        detect_path = temp_path
        display_name = "video_frame"
    else:
        if not os.path.exists(image_input):
            logger.error("File not found: %s", image_input)
            return []
        image_bgr = cv2.imread(image_input)
        if image_bgr is None:
            logger.error("Failed to decode image: %s", image_input)
            return []
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detect_path = image_input
        display_name = os.path.basename(image_input)

    try:
        # --- Step 0: Helmet detection -------------------------------------
        helmet_result = detect_helmets(detect_path)
        helmet_predictions = helmet_result.get("predictions", [])

        # --- Step 1: Detect plates with YOLOv8 ---------------------------
        yolo_result = detect_plates_yolov8(detect_path)
    finally:
        # Clean up temp file regardless of success/failure
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    predictions = yolo_result.get("predictions", [])

    # --- Summarise helmet detections --------------------------------------
    helmet_info: List[Dict[str, Any]] = []
    for hp in helmet_predictions:
        helmet_info.append({
            "class": hp.get("class", "unknown"),
            "confidence": hp.get("confidence", 0.0),
            "bbox": (
                max(int(hp["x"] - hp["width"] / 2), 0),
                max(int(hp["y"] - hp["height"] / 2), 0),
                int(hp["x"] + hp["width"] / 2),
                int(hp["y"] + hp["height"] / 2),
            ),
        })

    if not predictions and not helmet_predictions:
        logger.info("No license plates or helmet detections found.")
        return []

    # --- Step 2: Crop detected plate regions ------------------------------
    plate_crops = crop_plates_from_predictions(image_rgb, predictions)

    # --- Step 3: Read text from each crop with EasyOCR --------------------
    ocr_results = read_text_from_crops(plate_crops)

    # --- Build unified results list ---------------------------------------
    results: List[Dict[str, Any]] = []
    for crop_info, ocr in zip(plate_crops, ocr_results):
        avg_conf = float(np.mean(ocr["confidences"])) if ocr["confidences"] else 0.0
        plate_text = ocr["full_text"]
        has_valid_format = bool(
            INDIAN_PLATE_REGEX.search(plate_text.replace(" ", ""))
        ) if plate_text else False

        results.append({
            "plate_text": plate_text,
            "raw_text": ocr.get("raw_text", ""),
            "avg_confidence": avg_conf,
            "yolo_confidence": crop_info["confidence"],
            "valid_format": has_valid_format,
            "variant": ocr.get("variant", "?"),
            "texts": ocr["texts"],
            "confidences": ocr["confidences"],
            "crop_info": crop_info,
            "image_rgb": image_rgb,
            "helmet_detections": helmet_info,
        })

    # Filter out likely false-positive plate detections (too short)
    results = [
        r for r in results
        if not r["plate_text"]
        or INDIAN_PLATE_REGEX.search(r["plate_text"].replace(" ", ""))
        or len(r["plate_text"].replace(" ", "")) >= 7
    ]

    # If no plates found but helmets were detected, still return helmet info
    if not results and helmet_info:
        results.append({
            "plate_text": "",
            "raw_text": "",
            "avg_confidence": 0.0,
            "yolo_confidence": 0.0,
            "valid_format": False,
            "variant": "",
            "texts": [],
            "confidences": [],
            "crop_info": {},
            "image_rgb": image_rgb,
            "helmet_detections": helmet_info,
        })

    # --- Visualisation (optional) -----------------------------------------
    if show_visual and results:
        _visualise_results(image_rgb, results, display_name)

    return results


# ===========================================================================
#  Full Pipeline: Video (majority voting)
# ===========================================================================

def process_video(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> Optional[Dict[str, Any]]:
    """
    Sample *num_frames* evenly-spaced frames from a video, run the plate
    pipeline on each, and pick the best plate text via majority voting.

    Majority-voting score:
        ``vote_score = avg_per_frame_score + (vote_count − 1) × VOTE_FREQUENCY_BONUS``

    Args:
        video_path:  Path to the video file.
        num_frames:  Number of frames to sample.

    Returns:
        A result dict for the winning plate text, or ``None`` if no plates
        were detected in any frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0

    logger.info("Video: %d frames, %.1f FPS, %.1f s", total_frames, fps, duration)

    # Select frame indices (evenly spaced)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(
            0, total_frames - 1, num_frames, dtype=int
        ).tolist()

    logger.info("Sampling %d frames: %s", len(frame_indices), frame_indices)

    # --- Process every sampled frame --------------------------------------
    all_results: List[Tuple[float, Dict[str, Any], int]] = []

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning("Could not read frame %d — skipping.", frame_idx)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Frame %d/%d (index %d)", i + 1, len(frame_indices), frame_idx)

        frame_results = process_image(frame_rgb, show_visual=False)

        for res in frame_results:
            score = res["avg_confidence"] + res["yolo_confidence"]
            if res["valid_format"]:
                score += 2.0
            all_results.append((score, res, frame_idx))

    cap.release()

    if not all_results:
        logger.warning("No license plates detected in any sampled frame.")
        return None

    # --- Majority voting --------------------------------------------------
    vote_groups: Dict[str, List[Tuple[float, Dict[str, Any], int]]] = defaultdict(list)
    for score, res, frame_idx in all_results:
        vote_groups[res["plate_text"]].append((score, res, frame_idx))

    voted_results: List[Tuple[float, int, Tuple[float, Dict[str, Any], int], str]] = []
    for plate_text, entries in vote_groups.items():
        count = len(entries)
        avg_score = sum(s for s, _, _ in entries) / count
        best_entry = max(entries, key=lambda x: x[0])
        vote_score = avg_score + (count - 1) * VOTE_FREQUENCY_BONUS
        voted_results.append((vote_score, count, best_entry, plate_text))

    voted_results.sort(key=lambda x: x[0], reverse=True)
    best_vote_score, best_count, (best_score, best_result, best_frame_idx), best_text = (
        voted_results[0]
    )

    # --- Print summary ----------------------------------------------------
    _print_voting_summary(
        voted_results, all_results,
        best_text, best_count, best_vote_score, best_score,
        best_result, best_frame_idx, video_path,
    )

    # --- Visualise the winning frame --------------------------------------
    _visualise_best_frame(best_result, best_frame_idx, video_path)

    return best_result


# ===========================================================================
#  Save Annotated Image (for authority review)
# ===========================================================================

def save_annotated_image(
    image_rgb: np.ndarray,
    results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Draw detection bounding boxes, plate texts, and helmet boxes onto the
    image and save to *output_path*.  Returns the output path on success.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — no display needed

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title("SARAL — AI Detection Result")

    # Draw plate bounding boxes
    for res in results:
        if res.get("crop_info") and "bbox" in res["crop_info"]:
            x_min, y_min, x_max, y_max = res["crop_info"]["bbox"]
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            label = res["plate_text"] or "(no text)"
            ax.text(
                x_min, y_min - 8,
                f"{label}  ({res['yolo_confidence']:.0%})",
                color="yellow", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Draw helmet detection boxes
    helmet_drawn: set = set()
    for res in results:
        for hd in res.get("helmet_detections", []):
            bbox_key = hd["bbox"]
            if bbox_key in helmet_drawn:
                continue
            helmet_drawn.add(bbox_key)
            hx_min, hy_min, hx_max, hy_max = hd["bbox"]
            cls = hd["class"].lower()
            if "no" in cls or "without" in cls:
                edge_color = "orangered"
                label_prefix = "NO HELMET"
            else:
                edge_color = "lime"
                label_prefix = "HELMET"
            rect = patches.Rectangle(
                (hx_min, hy_min), hx_max - hx_min, hy_max - hy_min,
                linewidth=2, edgecolor=edge_color, facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                hx_min, hy_min - 8,
                f"{label_prefix} ({hd['confidence']:.0%})",
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=edge_color, alpha=0.7,
                ),
            )

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Annotated image saved to %s", output_path)
    return output_path


# ===========================================================================
#  Visualisation Helpers
# ===========================================================================

def _visualise_results(
    image_rgb: np.ndarray,
    results: List[Dict[str, Any]],
    display_name: str,
) -> None:
    """Show detected bounding boxes, plate texts, helmet detections, and cropped regions."""
    # --- Full image with bounding boxes ---
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"License Plate & Helmet Detection — {display_name}")

    # Draw plate bounding boxes
    for res in results:
        if res.get("crop_info") and "bbox" in res["crop_info"]:
            x_min, y_min, x_max, y_max = res["crop_info"]["bbox"]
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            label = res["plate_text"] or "(no text)"
            ax.text(
                x_min, y_min - 8,
                f"{label}  ({res['yolo_confidence']:.0%})",
                color="yellow", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Draw helmet detection bounding boxes (from the first result)
    helmet_drawn: set = set()
    for res in results:
        for hd in res.get("helmet_detections", []):
            bbox_key = hd["bbox"]
            if bbox_key in helmet_drawn:
                continue
            helmet_drawn.add(bbox_key)
            hx_min, hy_min, hx_max, hy_max = hd["bbox"]
            cls = hd["class"].lower()
            # Green for helmet, orange/red for no-helmet
            if "no" in cls or "without" in cls:
                edge_color = "orangered"
                label_prefix = "NO HELMET"
            else:
                edge_color = "lime"
                label_prefix = "HELMET"
            rect = patches.Rectangle(
                (hx_min, hy_min), hx_max - hx_min, hy_max - hy_min,
                linewidth=2, edgecolor=edge_color, facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                hx_min, hy_min - 8,
                f"{label_prefix} ({hd['confidence']:.0%})",
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=edge_color, alpha=0.7,
                ),
            )

    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # --- Console summary: Helmet Detections ---
    helmet_dets = []
    for res in results:
        for hd in res.get("helmet_detections", []):
            if hd not in helmet_dets:
                helmet_dets.append(hd)
    if helmet_dets:
        print("\nHELMET DETECTION RESULTS")
        print("-" * 40)
        for idx, hd in enumerate(helmet_dets):
            print(
                f"  Detection {idx + 1}: class='{hd['class']}'  "
                f"confidence={hd['confidence']:.2f}  "
                f"bbox={hd['bbox']}"
            )

    # --- Console summary: License Plates ---
    print("\nLICENSE PLATE OCR RESULTS")
    print("-" * 40)
    for idx, res in enumerate(results):
        if not res.get("crop_info"):
            continue
        print(f"\n  Plate {idx + 1}  (YOLO confidence: {res['yolo_confidence']:.2f})")
        if res["texts"]:
            for j, (text, conf) in enumerate(zip(res["texts"], res["confidences"])):
                print(f"    Segment {j + 1}: '{text}'  (OCR conf: {conf:.2f})")
            print(f"    Raw OCR text  : '{res['raw_text']}'")
            print(f"    Corrected text: '{res['plate_text']}'")
            print(f"    Best variant  : {res['variant']}")
        else:
            print("    No text detected by OCR.")

    # --- Individual crop images ---
    crops_with_data = [r for r in results if r.get("crop_info") and "crop" in r.get("crop_info", {})]
    n_crops = len(crops_with_data)
    if n_crops > 0:
        fig, axes = plt.subplots(1, n_crops, figsize=(6 * n_crops, 4))
        if n_crops == 1:
            axes = [axes]
        for i, res in enumerate(crops_with_data):
            axes[i].imshow(res["crop_info"]["crop"])
            title = res["plate_text"] or "(no text)"
            axes[i].set_title(f"Plate {i + 1}: '{title}'", fontsize=12)
            axes[i].axis("off")
        plt.suptitle("Cropped License Plate Regions", fontsize=14)
        plt.tight_layout()
        plt.show()


def _visualise_best_frame(
    best_result: Dict[str, Any],
    best_frame_idx: int,
    video_path: str,
) -> None:
    """Show the winning frame and its cropped plate from video processing."""
    image_rgb = best_result["image_rgb"]
    video_name = os.path.basename(video_path)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"Best Plate Detection — Frame {best_frame_idx} of {video_name}")

    x_min, y_min, x_max, y_max = best_result["crop_info"]["bbox"]
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)
    label = best_result["plate_text"] or "(no text)"
    ax.text(
        x_min, y_min - 8,
        f"{label}  ({best_result['yolo_confidence']:.0%})",
        color="yellow", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Cropped plate
    fig2, ax2 = plt.subplots(1, figsize=(6, 4))
    ax2.imshow(best_result["crop_info"]["crop"])
    ax2.set_title(f"Best Plate: '{best_result['plate_text']}'", fontsize=14)
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


def _print_voting_summary(
    voted_results, all_results,
    best_text, best_count, best_vote_score, best_score,
    best_result, best_frame_idx, video_path,
) -> None:
    """Print the majority-voting and per-frame detection summary tables."""
    separator = "=" * 60

    print(f"\n{separator}")
    print("BEST RESULT (majority voting across all sampled frames)")
    print(separator)
    print(f"  Plate text  : '{best_text}'")
    print(f"  Votes       : {best_count}/{len(all_results)} detections")
    print(f"  Vote score  : {best_vote_score:.2f}")
    print(f"  Best frame  : {best_frame_idx}")
    print(f"  Raw OCR text: '{best_result['raw_text']}'")
    print(f"  OCR conf    : {best_result['avg_confidence']:.2f}")
    print(f"  YOLO conf   : {best_result['yolo_confidence']:.2f}")
    print(f"  Valid format: {best_result['valid_format']}")
    print(f"  Variant     : {best_result['variant']}")
    print(f"  Frame score : {best_score:.2f}")

    # Ranked vote groups
    print(f"\n{separator}")
    print("MAJORITY VOTING RESULTS (ranked by vote score)")
    print(separator)
    for rank, (vscore, count, (iscore, ires, fidx), ptext) in enumerate(
        voted_results[:10], 1
    ):
        fmt = "VALID" if ires["valid_format"] else "invalid"
        print(
            f"  #{rank}  '{ptext}'  |  Votes: {count}  |  Vote score: {vscore:.2f}  "
            f"|  Best frame: {fidx}  |  {fmt}"
        )

    # Per-frame detections
    print(f"\n{separator}")
    print("ALL INDIVIDUAL DETECTIONS (ranked by per-frame score)")
    print(separator)
    sorted_results = sorted(all_results, key=lambda x: x[0], reverse=True)
    for rank, (score, res, f_idx) in enumerate(sorted_results[:10], 1):
        fmt = "VALID" if res["valid_format"] else "invalid"
        print(
            f"  #{rank}  Frame {f_idx:>4d}  |  '{res['plate_text']}'  "
            f"|  OCR: {res['avg_confidence']:.2f}  YOLO: {res['yolo_confidence']:.2f}  "
            f"|  {fmt}  |  Score: {score:.2f}"
        )


# ===========================================================================
#  CLI Entry-Point
# ===========================================================================

def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="saral_pretrained_colab",
        description=(
            "SARAL — Smart Automated Recognition of Automobile Licenses.  "
            "Detects and reads Indian license plates from images or videos."
        ),
    )
    parser.add_argument(
        "input_path",
        help="Path to an image or video file to process.",
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames to sample from a video (default: {DEFAULT_NUM_FRAMES}).",
    )
    return parser


def main(input_path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> None:
    """
    Top-level entry point: detect the input type and dispatch to
    ``process_image`` or ``process_video``.
    """
    separator = "=" * 60

    print(f"\n{separator}")
    print(f"PROCESSING: {input_path}")
    print(separator)

    if not os.path.exists(input_path):
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    input_type = get_input_type(input_path)
    logger.info("Detected input type: %s", input_type)

    if input_type == "image":
        results = process_image(input_path, show_visual=True)
        # Print a brief final summary
        print(f"\n{separator}")
        print("SUMMARY")
        print(separator)
        print(f"\nImage: {os.path.basename(input_path)}")

        # Helmet summary
        helmet_dets = []
        for res in results:
            for hd in res.get("helmet_detections", []):
                if hd not in helmet_dets:
                    helmet_dets.append(hd)
        if helmet_dets:
            no_helmet = [h for h in helmet_dets if "no" in h["class"].lower() or "without" in h["class"].lower()]
            with_helmet = [h for h in helmet_dets if h not in no_helmet]
            print(f"  Helmet detections: {len(with_helmet)} with helmet, {len(no_helmet)} without helmet")
        else:
            print("  Helmet detections: none")

        # Plate summary
        for j, res in enumerate(results):
            if res["texts"]:
                print(
                    f"  Plate {j + 1}: '{res['plate_text']}'  "
                    f"(Avg OCR confidence: {res['avg_confidence']:.2f})"
                )
            elif res.get("crop_info"):
                print(f"  Plate {j + 1}: No text detected")

    elif input_type == "video":
        process_video(input_path, num_frames=num_frames)


# ===========================================================================
#  *** CHANGE YOUR IMAGE/VIDEO PATH HERE ***
# ===========================================================================
if __name__ == "__main__":
    INPUT_PATH = "/content/1115.jpg"   # <-- Put your image or video path here
    NUM_FRAMES = 15                     # <-- Number of frames to sample (for video only)
    main(INPUT_PATH, num_frames=NUM_FRAMES)