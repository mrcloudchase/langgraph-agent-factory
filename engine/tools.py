"""Built-in tools — import and register any you need.

    from engine.tools import web_search, web_fetch, run_python
    tools.register(web_search, web_fetch, run_python)
"""

from __future__ import annotations

import sys
from io import StringIO

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information.
    Returns titles, URLs, and snippets for the most relevant results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )
    except Exception as exc:
        return f"Search error: {exc}"


@tool
def web_fetch(url: str) -> str:
    """Fetch and extract readable text from a URL.
    Strips navigation, scripts, and boilerplate. Returns up to 4000 chars."""
    try:
        response = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:4000]
    except Exception as exc:
        return f"Could not fetch {url}: {exc}"


@tool
def run_python(code: str) -> str:
    """Execute Python code and return its printed output.
    Use for data processing, calculations, formatting, or calling external APIs."""
    buf = StringIO()
    sys.stdout = buf
    try:
        exec(code, {})  # noqa: S102
        return buf.getvalue() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        sys.stdout = sys.__stdout__
