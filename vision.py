"""
vision.py — Image analysis pipeline.

Flow:
1. Quality check (blur + brightness). Reject bad images early with
   actionable feedback. This prevents garbage-in-garbage-out.
2. For maize: try the local PlantVillage model first (free, fast, offline).
3. If the local model is uncertain OR the crop is soybean, fall back to
   Gemini Vision with a structured "describe, don't diagnose" prompt.
4. Always return a list of described symptoms, never a final diagnosis.
   Diagnosis happens in the agent after retrieval — this keeps vision and
   knowledge grounded separately and auditable.
"""

import io
import logging
from typing import Optional, List, Dict

import cv2
import numpy as np
from PIL import Image

from llm import analyze_image as gemini_vision

logger = logging.getLogger(__name__)

# Quality thresholds tuned for field photos from phones. Adjust after testing.
BLUR_THRESHOLD = 80.0      # Laplacian variance; below this = too blurry
DARKNESS_THRESHOLD = 40    # Mean brightness 0-255; below this = too dark
BRIGHTNESS_THRESHOLD = 220 # Above this = overexposed

# Local model loaded lazily on first use. This avoids paying the startup
# cost if the user never uploads an image.
_local_model = None
_local_processor = None


def _load_local_model():
    global _local_model, _local_processor
    if _local_model is not None:
        return True
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        _local_processor = AutoImageProcessor.from_pretrained(model_id)
        _local_model = AutoModelForImageClassification.from_pretrained(model_id)
        _local_model.eval()
        logger.info("Loaded local PlantVillage model")
        return True
    except Exception as e:
        logger.warning("Could not load local vision model: %s", e)
        return False


def check_image_quality(image_bytes: bytes) -> Dict:
    """
    Returns {"ok": bool, "reason": str, "metrics": {...}}.
    If ok is False, reason is a farmer-friendly message.
    """
    try:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "reason": "Could not read the image file.", "metrics": {}}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())

        metrics = {"blur": round(blur_score, 1), "brightness": round(brightness, 1)}

        if blur_score < BLUR_THRESHOLD:
            return {
                "ok": False,
                "reason": "The photo is too blurry. Please hold the phone steady and retake, focusing on the affected leaf.",
                "metrics": metrics,
            }
        if brightness < DARKNESS_THRESHOLD:
            return {
                "ok": False,
                "reason": "The photo is too dark. Please retake in daylight or better lighting.",
                "metrics": metrics,
            }
        if brightness > BRIGHTNESS_THRESHOLD:
            return {
                "ok": False,
                "reason": "The photo is overexposed. Please retake out of direct harsh sunlight, ideally in shade.",
                "metrics": metrics,
            }
        return {"ok": True, "reason": "", "metrics": metrics}
    except Exception as e:
        logger.error("Image quality check failed: %s", e)
        return {"ok": False, "reason": "Could not process the image.", "metrics": {}}


def _run_local_model(image_bytes: bytes) -> Optional[Dict]:
    """Run the local PlantVillage model. Returns top prediction or None."""
    if not _load_local_model():
        return None
    try:
        import torch
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _local_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = _local_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        top_prob, top_idx = torch.max(probs, dim=-1)
        label = _local_model.config.id2label[int(top_idx)]
        return {
            "label": label,
            "confidence": float(top_prob),
            "source": "plantvillage_local",
        }
    except Exception as e:
        logger.warning("Local model inference failed: %s", e)
        return None


def _parse_plantvillage_label(label: str) -> Dict:
    """
    PlantVillage labels look like 'Corn_(maize)___Northern_Leaf_Blight' or
    'Corn_(maize)___healthy'. Parse into {crop, condition}.
    """
    parts = label.split("___")
    if len(parts) != 2:
        return {"crop": "unknown", "condition": label}
    crop_raw, condition_raw = parts
    crop = "maize" if "maize" in crop_raw.lower() or "corn" in crop_raw.lower() else crop_raw.lower()
    condition = condition_raw.replace("_", " ").lower()
    return {"crop": crop, "condition": condition}


def analyze_field_image(
    image_bytes: bytes,
    farmer_description: str,
    crop_hint: Optional[str] = None,
) -> Dict:
    """
    Main entry point. Returns:
    {
        "ok": bool,
        "quality_reason": str,    # populated if ok=False
        "symptoms": [str],        # described visible symptoms
        "candidate_conditions": [{"condition": str, "confidence": float}],
        "source": str,            # "plantvillage_local" | "gemini" | "none"
        "raw_description": str,   # freeform text for logging
    }
    """
    result = {
        "ok": False,
        "quality_reason": "",
        "symptoms": [],
        "candidate_conditions": [],
        "source": "none",
        "raw_description": "",
    }

    quality = check_image_quality(image_bytes)
    if not quality["ok"]:
        result["quality_reason"] = quality["reason"]
        return result

    # Route: try local model for maize, go straight to Gemini for soybean
    # since the local model doesn't cover soybean diseases well.
    use_local = crop_hint in (None, "maize") or not crop_hint

    if use_local:
        local = _run_local_model(image_bytes)
        if local and local["confidence"] >= 0.60:
            parsed = _parse_plantvillage_label(local["label"])
            if parsed["crop"] == "maize" or crop_hint == "maize":
                result["ok"] = True
                result["source"] = "plantvillage_local"
                result["candidate_conditions"] = [{
                    "condition": parsed["condition"],
                    "confidence": local["confidence"],
                }]
                result["symptoms"] = [parsed["condition"]]
                result["raw_description"] = f"Local model: {parsed['condition']} ({local['confidence']:.2f})"
                return result
            # Local model returned a non-maize prediction; fall through to Gemini.

    # Gemini Vision fallback. Critical: ask it to DESCRIBE, not diagnose.
    vision_prompt = f"""You are analyzing a field photograph of a crop for an agricultural advisory system.

Crop context: {crop_hint or "unknown, possibly maize or soybean"}
Farmer's description: {farmer_description or "not provided"}

Your task: DESCRIBE what you see. Do NOT diagnose.

List visible symptoms only. Focus on:
- Leaf color (yellowing, browning, purpling, paleness)
- Spots, lesions, or holes (size, color, pattern, distribution)
- Wilting or leaf rolling
- Growth issues (stunting, uneven growth)
- Pests visible (insects, larvae, webbing)

Return valid JSON only, no markdown:
{{
  "symptoms": ["symptom 1", "symptom 2", ...],
  "crop_visible": "maize" | "soybean" | "other" | "unknown",
  "image_clear_enough": true | false,
  "notes": "one sentence of context, or empty string"
}}"""

    raw = gemini_vision(image_bytes, vision_prompt)
    if not raw:
        result["quality_reason"] = "Image analysis service is unavailable. Please try again or describe the issue in text."
        return result

    # Strip any accidental markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        import json
        parsed = json.loads(cleaned)
        result["ok"] = True
        result["source"] = "gemini"
        result["symptoms"] = parsed.get("symptoms", [])
        result["raw_description"] = parsed.get("notes", "") or raw
        # Gemini describes, doesn't classify, so candidate_conditions stays empty.
    except Exception as e:
        logger.warning("Could not parse Gemini vision JSON: %s", e)
        result["ok"] = True
        result["source"] = "gemini"
        result["raw_description"] = raw
        result["symptoms"] = [raw[:200]]  # fallback: treat whole response as one symptom blob

    return result
