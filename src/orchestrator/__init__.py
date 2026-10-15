"""
Scrub-Voice Central Orchestration Package.

Dispatches validated voice-intent commands between NLU and downstream subsystems.
"""

from .dispatcher import CommandDispatcher

__all__ = ["CommandDispatcher"]
