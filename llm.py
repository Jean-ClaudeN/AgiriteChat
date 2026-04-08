"""
llm.py — Groq wrapper for both text and vision.

Why Groq:
- Truly free tier with generous rate limits (no billing/credit card required)
- Very fast inference (LPU-based)
- OpenAI-compatible API, clean Python SDK
- Good quality Llama 3.3 70B for text, Llama 4 Scout 17B for vision

Models used:
- Text: llama-3.3-70b-versatile (strong reasoning, good for structured JSON output)
- Vision: meta-llama/llama-4-scout-17b-16e-instruct (Llama 3.2 Vision Preview is
  deprecated; Llama 4 Scout is the current vision model on Groq)
"""

import os
import json
import time
import base64
import logging
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_client: Optional[Groq] = None


def _get_client() -> Optional[Groq]:
    """Lazy-init a Groq client from env var or Streamlit secrets."""
    global _client
    if _client is not None:
        return _client

    key = os.getenv("GROQ_API_KEY")
    if not key:
        # Streamlit secrets fallback
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if not key:
        logger.warning("GROQ_API_KEY not found; LLM calls will fail.")
        return None

    try:
        _client = Groq(api_key=key)
        return _client
    except Exception as e:
        logger.error("Failed to create Groq client: %s", e)
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
    """Call Groq text model. Returns None on hard failure."""
    client = _get_client()
    if client is None:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            logger.warning("Groq text returned empty response (attempt %d)", attempt + 1)
        except Exception as e:
            logger.warning("Groq text call failed (attempt %d): %s", attempt + 1, e)
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return None


def generate_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[dict]:
    """Call Groq asking for JSON and parse it. Returns None on failure."""
    client = _get_client()
    if client is None:
        return None

    messages = []
    if system:
        # Groq needs "JSON" mentioned in the prompt when using response_format json_object
        messages.append({"role": "system", "content": system + " Respond only with valid JSON."})
    else:
        messages.append({"role": "system", "content": "Respond only with valid JSON."})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            logger.warning("Groq JSON returned empty response")
            return None
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Groq returned invalid JSON: %s", e)
        return None
    except Exception as e:
        logger.warning("Groq JSON call failed: %s", e)
        return None


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect image mime type from magic bytes. Falls back to JPEG."""
    if not image_bytes or len(image_bytes) < 8:
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] in (b"GIF8",):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def analyze_image(image_bytes: bytes, prompt: str) -> Optional[str]:
    """
    Send an image to Groq Vision. Returns a plain-text description.

    Groq's vision API is OpenAI-compatible: the image goes in as a
    base64-encoded data URL inside the message content.
    """
    client = _get_client()
    if client is None:
        logger.warning("Groq vision called but client unavailable")
        return None

    if not image_bytes:
        logger.warning("Groq vision called with empty image bytes")
        return None

    mime_type = _detect_mime_type(image_bytes)
    logger.info("Groq vision: %d bytes, mime=%s", len(image_bytes), mime_type)

    try:
        # Encode image as base64 data URL
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            logger.warning("Groq vision returned empty; finish_reason=%s",
                           resp.choices[0].finish_reason if resp.choices else "unknown")
            return None
        return text
    except Exception as e:
        logger.warning("Groq vision call failed: %s", e)
        return None
