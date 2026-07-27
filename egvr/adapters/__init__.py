"""Optional backend adapters for EGVR-Agent."""

from .tool_server import ToolServerClient, ToolServerError

__all__ = ["ToolServerClient", "ToolServerError"]
