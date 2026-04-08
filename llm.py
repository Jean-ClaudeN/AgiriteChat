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
