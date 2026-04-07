"""
llm.py — Gemini wrapper using the new google-genai SDK.

Why this exists:
- Centralizes all LLM calls so retries, model choice, and prompt handling
  live in one place.
- Keeps agent.py and vision.py free of API-key handling and SDK details.

Uses the google-genai package (NOT the deprecated google-generativeai).
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
        # Streamlit secrets fallback (lazy import so module works outside Streamlit)
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
            return (resp.text or "").strip()
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
        raw = (resp.text or "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Gemini returned invalid JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("Gemini JSON call failed: %s", e)
        return None


def analyze_image(image_bytes: bytes, prompt: str) -> Optional[str]:
    """Send an image to Gemini Vision. Returns a plain-text description."""
    client = _get_client()
    if client is None:
        return None

    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
        return (resp.text or "").strip()
    except Exception as e:
        logger.warning("Gemini vision call failed: %s", e)
        return None
