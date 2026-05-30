from .chain import ChainBuilder
from .evaluator import EvaluatorBuilder
from .llm import LlmBuilder
from .orchestrator import OrchestratorBuilder
from .parallel import ParallelBuilder
from .react import ReactBuilder
from .router import RouterBuilder

BUILDERS: dict[str, type] = {
    # ── Leaf types (nodes) ──────────────────────────────────────────────────
    "llm":          LlmBuilder,    # single LLM call
    "react":        ReactBuilder,  # LLM-controlled tool loop
    # ── Workflow types (topology) ───────────────────────────────────────────
    "chain":        ChainBuilder,
    "router":       RouterBuilder,
    "parallel":     ParallelBuilder,
    "orchestrator": OrchestratorBuilder,
    "evaluator":    EvaluatorBuilder,
}
