"""Built-in tools provided by the factory.

All four are available to any agent by name:

    tools:
      - web_search
      - web_fetch
      - bash
      - run_python

Register them all at once:

    from engine.tools import BUILTIN_TOOLS
    tools.register(*BUILTIN_TOOLS)
"""

from __future__ import annotations

import subprocess

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool

_TIMEOUT = 30


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
        resp = httpx.get(
            url, timeout=_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:4000]
    except Exception as exc:
        return f"Could not fetch {url}: {exc}"


@tool
def bash(command: str) -> str:
    """Execute a bash command and return its output.
    Use for file operations, system commands, or running installed CLIs."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=_TIMEOUT,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return (f"{out}\nError:\n{err}" if out else f"Error:\n{err}").strip()
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_TIMEOUT}s"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def run_python(code: str) -> str:
    """Execute Python code via `python -c` and return its output.
    Runs in an isolated subprocess — safe for data processing, calculations,
    API calls, or anything that needs the full Python stdlib."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return (f"{out}\nError:\n{err}" if out else f"Error:\n{err}").strip()
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: code timed out after {_TIMEOUT}s"
    except Exception as exc:
        return f"Error: {exc}"


BUILTIN_TOOLS = [web_search, web_fetch, bash, run_python]
