from __future__ import annotations

from typing import Any

from langchain_core.tools import tool as _lc_tool


class ToolRegistry:
    """Holds the tools available to agents.

    Use .register() for pre-decorated @tool functions and .tool as a
    decorator for plain functions you want to turn into tools inline.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, *fns: Any) -> None:
        """Register one or more LangChain @tool-decorated functions."""
        for fn in fns:
            self._tools[fn.name] = fn

    def tool(self, fn: Any) -> Any:
        """Decorator: wrap fn as a LangChain tool and register it."""
        wrapped = _lc_tool(fn)
        self._tools[wrapped.name] = wrapped
        return wrapped

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not registered. Available: {self.names}"
            )
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)
