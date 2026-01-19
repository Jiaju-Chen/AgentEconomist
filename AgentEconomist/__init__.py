"""
Agent Economist - Restructured Version

A modular, framework-agnostic economic simulation research agent.
"""

__version__ = "2.0.0"

# Core modules (framework-independent)
from . import utils
from . import core
from . import state
from . import tools
from . import prompts

# LangGraph-specific (optional import)
try:
    from . import graph
    from .graph import build_economist_graph
    __all__ = [
        "utils",
        "core",
        "state",
        "tools",
        "prompts",
        "graph",
        "build_economist_graph",
    ]
except ImportError as e:
    # LangGraph not available
    print(f"Warning: LangGraph modules not available: {e}")
    __all__ = [
        "utils",
        "core",
        "state",
        "tools",
        "prompts",
    ]
