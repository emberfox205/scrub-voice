"""
Monitor Intent Handlers Package.
Exports handler registry and imports handler modules to trigger registration.
"""

from src.monitor.handlers.registry import HANDLERS, handle_command, register_intent

# Import handler modules to execute decorators
from src.monitor.handlers import scan_handlers
from src.monitor.handlers import event_handlers
from src.monitor.handlers import emergency_handlers
from src.monitor.handlers import tool_handlers

__all__ = [
    "HANDLERS",
    "handle_command",
    "register_intent",
    "scan_handlers",
    "event_handlers",
    "emergency_handlers",
    "tool_handlers",
]
