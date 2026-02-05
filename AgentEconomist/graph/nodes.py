"""
LangGraph 节点定义

定义 Agent 工作流中的各个节点。
"""

import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import AgentState
from ..state.converter import update_fs_state_from_manifest
from ..utils.response import extract_manifest_path


def create_init_node(system_prompt: str):
    """创建初始化节点"""
    def init_node(state: AgentState):
        """初始化节点：添加系统提示，并清理中断残留的状态"""
        messages = state.get("messages", [])
        updates = {}
        
        # 如果还没有系统消息，则插入系统提示
        if not any(isinstance(m, SystemMessage) for m in messages):
            updates["messages"] = [SystemMessage(content=system_prompt), *messages]
        
        # 重要：检测到新的用户消息时，清除可能残留的 running_tool_name
        # 这样可以避免用户在工具执行期间点击暂停时，工具名称一直显示的 bug
        if messages and isinstance(messages[-1], HumanMessage):
            current_tool = state.get("running_tool_name")
            if current_tool:
                # 有新的用户消息，清除之前可能因中断而残留的工具名
                updates["running_tool_name"] = None
        
        return updates if updates else {}
    return init_node


def create_call_model_node(llm_with_tools, system_prompt: str):
    """创建模型调用节点"""
    def call_model(state: AgentState):
        """调用 LLM，智能清理消息序列避免 API 错误"""
        messages = state.get("messages", [])
        
        # 策略：只保留完整的对话轮次，移除不完整的工具调用序列
        # 1. 提取系统消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        
        # 2. 提取非系统消息
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # 3. 智能过滤：保留完整对话历史，但移除孤立的 ToolMessage
        clean_msgs = []
        i = 0
        while i < len(other_msgs):
            msg = other_msgs[i]
            
            if isinstance(msg, HumanMessage):
                # 保留所有 Human 消息
                clean_msgs.append(msg)
                i += 1
            
            elif isinstance(msg, AIMessage):
                # 保留 AI 消息
                clean_msgs.append(msg)
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                i += 1
                
                # 如果 AI 有 tool_calls，保留紧随其后的 ToolMessage
                if has_tool_calls:
                    while i < len(other_msgs) and isinstance(other_msgs[i], ToolMessage):
                        clean_msgs.append(other_msgs[i])
                        i += 1
            
            elif isinstance(msg, ToolMessage):
                # 孤立的 ToolMessage（前面没有 AI tool_calls），跳过
                i += 1
            
            else:
                # 其他类型消息，保留
                clean_msgs.append(msg)
                i += 1
        
        # 4. 组合：系统消息 + 清理后的消息
        if system_msgs:
            final_msgs = [system_msgs[0], *clean_msgs]
        else:
            final_msgs = [SystemMessage(content=system_prompt), *clean_msgs]
        
        # 调用模型
        response = llm_with_tools.invoke(final_msgs)
        
        # 检测工具调用，设置 running_tool_name
        tool_name = None
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_name = response.tool_calls[0]['name']
        
        # 追加模型回复和工具名称
        # 注意：这里会清除之前的 running_tool_name（如果没有新的工具调用）
        # 这样即使中断发生，下次调用时也会清理掉残留的工具名
        return {
            "messages": [response],
            "running_tool_name": tool_name  # None 或新的工具名
        }
    
    return call_model


def create_update_fs_state_node():
    """创建状态更新节点"""
    def update_fs_state_node(state: AgentState):
        """
        工具调用后更新 fs_state
        
        关键逻辑：
        1. 遍历所有 ToolMessage，寻找最新的 manifest_path
        2. 调用 update_fs_state_from_manifest() 转换 FSState
        3. 更新状态
        """
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        # 遍历所有 ToolMessage（从最新到最旧），寻找 manifest_path
        manifest_path = None
        last_tool_output = None
        
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                continue
            
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            
            # 尝试提取 manifest_path
            path = extract_manifest_path(content)
            if path:
                manifest_path = path
                last_tool_output = content
                break
        
        if not manifest_path:
            # 没有找到 manifest_path，不更新状态
            return {}
        
        try:
            # 更新 FSState
            fs_state = update_fs_state_from_manifest(manifest_path)
            
            return {
                "fs_state": fs_state,
                "manifest_path": manifest_path,
                "last_tool_output": last_tool_output,
                "running_tool_name": None  # 工具执行完毕，清空
            }
        except Exception as e:
            print(f"Warning: Failed to update fs_state from manifest {manifest_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "running_tool_name": None  # 即使失败也清空
            }
    
    return update_fs_state_node
