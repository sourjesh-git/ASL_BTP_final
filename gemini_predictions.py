"""
Gemini-powered word completion for ASL recognition.
Uses prefix from CNN output, calls Gemini API for completions, caches results.
"""
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# In-memory cache: prefix (lowercase) -> list of completion strings
_cache = {}

GEMINI_MODEL = "gemini-2.5-flash"
NUM_COMPLETIONS = 5


def _get_client():
    from google import genai
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
    return genai.Client(api_key=api_key)


def get_word_completions(prefix: str, top_k: int = 5) -> list:
    """
    Returns up to top_k word completions for prefix.
    Only calls API when len(prefix) >= 2; otherwise returns [].
    Uses and updates internal cache.
    """
    prefix = prefix.strip()
    if len(prefix) < 2:
        return []
    key = prefix.lower()
    if key in _cache:
        return _cache[key][:top_k]
    try:
        client = _get_client()
        prompt = (
            f'Given the prefix "{prefix}", list exactly {NUM_COMPLETIONS} English words '
            'that start with this prefix. Return ONLY a JSON array of strings, no other text. '
            'Example: ["Word1", "Word2", "Word3", "Word4", "Word5"]'
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if "```" in text:
            text = re.sub(r"^```\w*\n?", "", text).strip()
            text = re.sub(r"\n?```$", "", text).strip()
        words = json.loads(text)
        if isinstance(words, list):
            words = [str(w).strip() for w in words if w][:NUM_COMPLETIONS]
        else:
            words = []
        _cache[key] = words
        return words[:top_k]
    except Exception as e:
        print("Gemini prediction error:", e)
        return []
