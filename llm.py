"""
llm.py — Gemini wrapper using the new google-genai SDK.

Centralizes all LLM calls. Uses the google-genai package (NOT the
deprecated google-generativeai).
"""

import os
import json
import time
import logging
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Gemini 2.0 Flash: free tier, fast, supports vision and JSON mode.
TEXT_MODEL = "gemini-2.0-flash"
VISION_MODEL = "gemini-2.0-flash"

_client: Optional[genai.Client] = None


def _get_client() -> Optional[genai.Client]:
    """Lazy-init a Gemini client from env var or Streamlit secrets."""
    global _client
    if _client is not None:
        return _client

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not key:
        logger.warning("GEMINI_API_KEY not found; LLM calls will fail.")
        return None

    try:
        _client = genai.Client(api_key=key)
        return _client
    except Exception as e:
        logger.error("Failed to create Gemini client: %s", e)
        return None


def is_available() -> bool:
    return _get_client() is not None


def _extract_text(resp) -> str:
    """
    Defensively extract text from a Gemini response.
    The .text property can return None or raise if the response was
    blocked by safety filters or has an unusual structure. This helper
    walks candidates and parts manually to get whatever text exists.
    """
    if resp is None:
        return ""

    # Try the easy path first
    try:
        t = resp.text
        if t:
            return t
    except Exception:
        pass

    # Walk candidates and parts manually
    try:
        candidates = getattr(resp, "candidates", None) or []
        pieces = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for p in parts:
                text = getattr(p, "text", None)
                if text:
                    pieces.append(text)
        return "".join(pieces)
    except Exception as e:
        logger.warning("Could not extract text from Gemini response: %s", e)
        return ""


def generate_text(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> Optional[str]:
    """Call Gemini text model. Returns None on hard failure."""
    client = _get_client()
    if client is None:
        return None

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=1024,
        system_instruction=system,
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=config,
            )
            text = _extract_text(resp).strip()
            if text:
                return text
            logger.warning("Gemini text returned empty response (attempt %d)", attempt + 1)
        except Exception as e:
            logger.warning("Gemini text call failed (attempt %d): %s", attempt + 1, e)
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return None


def generate_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[dict]:
    """Call Gemini asking for JSON and parse it. Returns None on failure."""
    client = _get_client()
    if client is None:
        return None

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=1024,
        system_instruction=system,
        response_mime_type="application/json",
    )

    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config=config,
        )
        raw = _extract_text(resp).strip()
        if not raw:
            logger.warning("Gemini JSON returned empty response")
            return None
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Gemini returned invalid JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("Gemini JSON call failed: %s", e)
        return None


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect image mime type from magic bytes. Falls back to JPEG."""
    if not image_bytes or len(image_bytes) < 8:
        return "image/jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # JPEG: FF D8 FF
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # GIF: 47 49 46 38
    if image_bytes[:4] in (b"GIF8",):
        return "image/gif"
    # WebP: 52 49 46 46 ... 57 45 42 50
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    # Default: let Gemini try to figure it out as JPEG
    return "image/jpeg"


def analyze_image(image_bytes: bytes, prompt: str) -> Optional[str]:
    """Send an image to Gemini Vision. Returns a plain-text description."""
    client = _get_client()
    if client is None:
        logger.warning("Gemini vision called but client unavailable")
        return None

    if not image_bytes:
        logger.warning("Gemini vision called with empty image bytes")
        return None

    mime_type = _detect_mime_type(image_bytes)
    logger.info("Gemini vision: %d bytes, mime=%s", len(image_bytes), mime_type)

    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
        text = _extract_text(resp).strip()
        if not text:
            # Log the finish reason for debugging
            try:
                candidates = getattr(resp, "candidates", None) or []
                if candidates:
                    reason = getattr(candidates[0], "finish_reason", "unknown")
                    logger.warning("Gemini vision returned empty; finish_reason=%s", reason)
                    safety = getattr(candidates[0], "safety_ratings", None)
                    if safety:
                        logger.warning("Gemini vision safety ratings: %s", safety)
            except Exception:
                pass
            return None
        return text
    except Exception as e:
        logger.warning("Gemini vision call failed: %s", e)
        return None
