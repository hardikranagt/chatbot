"""
Web search via the Serper API (https://serper.dev).

Set the SERPER_API_KEY environment variable (or add it to .env).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"


def search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Run a Google search via Serper and return a list of results.
    Each result dict has: title, link, snippet.
    Returns an empty list if the API key is missing or the call fails.
    """
    if not SERPER_API_KEY:
        return []

    try:
        response = requests.post(
            SERPER_URL,
            json={"q": query, "num": num_results},
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


def format_web_results(results: list[dict]) -> str:
    """Format web search results into a text block for the LLM prompt."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n    {r['snippet']}\n    Source: {r['link']}")
    return "\n\n".join(parts)
