"""General-purpose tools that service agents can use.

These tools are domain-agnostic — they give agents the ability to reach
the open web, process data, and deliver output to any destination.
That combination covers the vast majority of possible services without
needing domain-specific integrations.

In production, deliver_output would call real email/Slack/webhook APIs.
Here it prints to stdout so the demo is fully visible.
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
def browse_url(url: str) -> str:
    """Fetch and extract readable text from a URL.
    Strips navigation, scripts, and boilerplate. Returns up to 4000 chars."""
    try:
        response = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
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
    Use for data processing, number crunching, or formatting results."""
    buf = StringIO()
    sys.stdout = buf
    try:
        exec(code, {})  # noqa: S102
        return buf.getvalue() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        sys.stdout = sys.__stdout__


@tool
def deliver_output(destination: str, content: str) -> str:
    """Deliver the final output to the customer's chosen destination.

    destination formats:
      email:address@domain.com
      slack:#channel-name
      webhook:https://...
    """
    width = 64
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  Delivered → {destination}")
    print(bar)
    print(content)
    print(f"{bar}\n")
    return f"Delivered to {destination}"


TOOL_REGISTRY: dict[str, object] = {
    t.name: t for t in [web_search, browse_url, run_python, deliver_output]
}
