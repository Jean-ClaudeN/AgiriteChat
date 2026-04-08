"""
vision.py — Image analysis pipeline.

Flow:
1. Quality check (blur + brightness). Reject bad images early.
2. Try Gemini Vision first (it's always available on Streamlit Cloud and
   handles both maize and soybean). Ask it to DESCRIBE, not diagnose.
3. The local PlantVillage model is DISABLED by default on Streamlit Cloud
   because transformers downloads often fail in sandboxed environments
   and push memory over the free-tier limit. To enable locally, set the
   env var ENABLE_LOCAL_VISION_MODEL=1.
4. Always return described symptoms, never a final diagnosis. Diagnosis
   happens in the agent after retrieval — this keeps vision and knowledge
   grounded separately and auditable.
"""

import io
import os
import json
import logging
from typing import Optional, List, Dict

import cv2
import numpy as np
from PIL import Image

from llm import analyze_image as gemini_vision

logger = logging.getLogger(__name__)

# Quality thresholds tuned for field photos from phones.
BLUR_THRESHOLD = 60.0      # Lowered from 80 — less aggressive rejection
DARKNESS_THRESHOLD = 30    # Lowered from 40
BRIGHTNESS_THRESHOLD = 230 # Raised from 220

# Local model disabled by default. Set ENABLE_LOCAL_VISION_MODEL=1 to enable.
LOCAL_MODEL_ENABLED = os.getenv("ENABLE_LOCAL_VISION_MODEL", "0") == "1"

_local_model = None
_local_processor = None


def _load_local_model() -> bool:
    """Lazy-load the PlantVillage model. Returns False if unavailable."""
    global _local_model, _local_processor
    if not LOCAL_MODEL_ENABLED:
        return False
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
    """Returns {'ok': bool, 'reason': str, 'metrics': {...}}."""
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
                "reason": "The photo appears blurry. Please retake with a steady hand, focused on the affected leaf.",
                "metrics": metrics,
            }
        if brightness < DARKNESS_THRESHOLD:
            return {
                "ok": False,
                "reason": "The photo is very dark. Please retake in daylight or better lighting.",
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
        return {"ok": False, "reason": f"Could not process the image: {e}", "metrics": {}}


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
    """PlantVillage labels look like 'Corn_(maize)___Northern_Leaf_Blight'."""
    parts = label.split("___")
    if len(parts) != 2:
        return {"crop": "unknown", "condition": label}
    crop_raw, condition_raw = parts
    crop = "maize" if "maize" in crop_raw.lower() or "corn" in crop_raw.lower() else crop_raw.lower()
    condition = condition_raw.replace("_", " ").lower()
    return {"crop": crop, "condition": condition}


def _extract_json_loosely(raw: str) -> Optional[dict]:
    """Try hard to extract JSON from an LLM response that may have fences or prose."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("```", 1)[1]
        if text.lower().startswith("json"):
            text = text[4:]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()

    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { and last } and try that slice
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def analyze_field_image(
    image_bytes: bytes,
    farmer_description: str,
    crop_hint: Optional[str] = None,
) -> Dict:
    """
    Main entry point. Returns:
    {
        "ok": bool,
        "quality_reason": str,
        "symptoms": [str],
        "candidate_conditions": [{"condition": str, "confidence": float}],
        "source": str,
        "raw_description": str,
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

    # Step 1: Quality check
    quality = check_image_quality(image_bytes)
    if not quality["ok"]:
        result["quality_reason"] = quality["reason"]
        return result

    # Step 2: Try local model if enabled (default: disabled on Streamlit Cloud)
    if LOCAL_MODEL_ENABLED and crop_hint in (None, "maize", "general"):
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

    # Step 3: Gemini Vision (primary path in production)
    vision_prompt = f"""You are analyzing a field photograph for an agricultural advisory system for maize and soybean farmers.

Crop context: {crop_hint or "unknown, possibly maize or soybean"}
Farmer's description: {farmer_description or "not provided"}

Your task: DESCRIBE what you see. Do NOT diagnose any disease.

List visible observations. Focus on:
- Leaf color (yellowing, browning, purpling, paleness, interveinal patterns)
- Spots, lesions, or holes (size, color, pattern, distribution)
- Wilting, curling, or leaf rolling
- Growth issues (stunting, uneven growth)
- Visible pests (insects, larvae, webbing, frass)
- If the photo is a WIDE shot of a whole field with no clear close-up, say so

Return ONLY valid JSON, no markdown, no prose:
{{
  "symptoms": ["observation 1", "observation 2"],
  "crop_visible": "maize" | "soybean" | "other" | "unknown",
  "image_clear_enough": true,
  "notes": "one sentence of helpful context"
}}

If the image is too wide-angle to see plant details, set image_clear_enough to false and explain in notes."""

    raw = gemini_vision(image_bytes, vision_prompt)

    if not raw:
        result["quality_reason"] = (
            "The AI vision service did not return a response. "
            "This can happen with very large images or temporary service issues. "
            "Please try a smaller, close-up photo of the affected leaf, "
            "or describe the issue in the Ask tab."
        )
        return result

    # Try to parse JSON; if parsing fails, still return the raw text as a symptom
    parsed = _extract_json_loosely(raw)

    if parsed:
        result["ok"] = True
        result["source"] = "gemini"
        symptoms = parsed.get("symptoms", [])
        if isinstance(symptoms, list):
            result["symptoms"] = [str(s) for s in symptoms if s]
        result["raw_description"] = parsed.get("notes", "") or raw[:300]

        # If Gemini says the image isn't clear enough, pass that to the farmer
        if parsed.get("image_clear_enough") is False:
            note = parsed.get("notes", "")
            result["quality_reason"] = (
                f"The photo is not clear enough for detailed analysis. {note} "
                "Please upload a close-up of an affected leaf."
            )
            # Still return ok=True so the agent can try to help with what it saw
    else:
        # JSON parse failed — use the raw text as best we can
        result["ok"] = True
        result["source"] = "gemini"
        result["raw_description"] = raw[:500]
        # Treat the whole response as one big symptom blob
        result["symptoms"] = [raw[:200]]
        logger.warning("Gemini vision returned non-JSON, using raw text")

    return result
