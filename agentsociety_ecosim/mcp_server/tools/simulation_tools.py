"""Stub simulation tool registrations for MCP server."""
from typing import Any
import logging

logger = logging.getLogger(__name__)


def register_tools(mcp: Any, parameter_manager: Any) -> None:
    """Register simulation-related tools with MCP, if available."""
    if mcp is None:
        logger.warning("MCP instance not provided; skipping simulation tool registration")
        return
    # 在没有具体工具实现时，留空即可，防止导入失败导致服务器启动中断
    logger.info("simulation_tools.register_tools invoked, but no tools to register")
