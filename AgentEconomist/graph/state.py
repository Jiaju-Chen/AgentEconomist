"""
LangGraph State 定义

定义 LangGraph 使用的状态结构（包装 FSState + messages）。
"""

from typing import List, Annotated, Optional
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict  # Python 3.12+
from langgraph.graph.message import add_messages

from ..state.types import FSState, create_empty_fs_state


class AgentState(TypedDict):
    """
    LangGraph Agent 的完整状态
    
    包含:
    - messages: LangChain 消息列表（自动累积）
    - fs_state: 前端需要的 FSState
    - manifest_path: 当前实验的 manifest 路径
    - last_tool_output: 最后一次工具调用的输出
    """
    messages: Annotated[List[dict], add_messages]  # 自动累积消息
    fs_state: FSState
    manifest_path: Optional[str]
    last_tool_output: Optional[str]


def create_initial_state() -> AgentState:
    """
    创建初始的 Agent 状态
    
    Returns:
        初始化的 AgentState
    """
    return AgentState(
        messages=[],
        fs_state=create_empty_fs_state(),
        manifest_path=None,
        last_tool_output=None
    )
