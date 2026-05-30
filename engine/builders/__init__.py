from .chain import ChainBuilder
from .evaluator import EvaluatorBuilder
from .orchestrator import OrchestratorBuilder
from .parallel import ParallelBuilder
from .react import ReactBuilder
from .router import RouterBuilder
from .swarm import SwarmBuilder

BUILDERS: dict[str, type] = {
    "react":        ReactBuilder,
    "chain":        ChainBuilder,
    "router":       RouterBuilder,
    "parallel":     ParallelBuilder,
    "orchestrator": OrchestratorBuilder,
    "evaluator":    EvaluatorBuilder,
    "swarm":        SwarmBuilder,
}
