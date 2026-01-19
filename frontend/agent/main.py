# 1. 导入必要库
import asyncio
import time
import json
import uuid
from pathlib import Path
from typing import Annotated, TypedDict, List, Optional, Dict

from copilotkit import CopilotKitState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from copilotkit.langgraph import copilotkit_emit_state

# 导入核心模型与消息类
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import FakeListChatModel

# 定义嵌套的 TypedDict 类
class KnowledgeBaseItem(TypedDict):
    """文献支撑项"""
    title: str                    # 文献标题
    source: str                   # 出处（期刊名 + 年份，如 "Nature 2023"）
    url: Optional[str]            # 论文链接（PDF/DOI链接）
    core_finding: str             # 核心发现（1-2句话总结）
    takeaway: str                 # 对本研究的启示

class ConfigurationItem(TypedDict):
    """实验配置项"""
    filename: str                 # treatmentA.yaml
    content: str                  # 修改后的yaml文件内容

class ImageItem(TypedDict):
    """实验结果图片项"""
    name: str                     # 图片名称/描述
    data: str                     # 图片的Base64编码/SVG内容

class PathsDict(TypedDict):
    """文件路径字典"""
    experiment_directory: str    # 实验目录路径（相对路径）
    manifest_file: str            # manifest.yaml 路径（相对路径）
    config_files: Dict[str, str]  # 组名: 配置文件路径（相对路径）

class FSState(TypedDict):
    """Field Simulation Agent 状态结构"""
    # ========== 实验基本信息 ==========
    experiment_id: str            # 实验唯一标识符
    name: str                     # 实验名称
    description: str              # 实验描述
    created_date: str             # 创建日期 (ISO 8601: "2026-01-01")
    tags: List[str]               # 标签列表
    
    # ========== 1. 文献支撑 ==========
    knowledge_base: List[KnowledgeBaseItem]  # 文献支撑列表
    
    # ========== 实验状态 ==========
    status: str                   # 整体状态: "pending" | "running" | "completed" | "failed" | "analysis_pending"
    start_time: Optional[str]     # 开始时间 (ISO 8601)
    end_time: Optional[str]       # 结束时间 (ISO 8601)
    duration_seconds: Optional[float]  # 总耗时（秒）
    
    # ========== 2. 研究想法 ==========
    research_question: str        # 研究问题
    hypothesis: str              # 研究假设
    expected_outcome: str         # 预期结果
    
    # ========== 4. 实验配置 ==========
    configurations: List[ConfigurationItem]  # 实验配置列表
    
    # ========== 实验结果 ==========
    images: List[ImageItem]       # 实验结果图片列表
    
    # ========== 5. 文件路径 ==========
    paths: PathsDict             # 文件路径字典

# 2. 定义Agent状态（这是LangGraph工作流的核心）
class AgentState(CopilotKitState):
    """定义Agent的状态结构"""
    num_input: int
    test_num: int
    logs: List[str]
    fs_state: FSState
    running_tool_name: str
    session_index: int  # 当前session索引
    processed_tool_count: int  # 当前session中已处理的tool数量

# 3. 创建模拟模型 - 定义你期望的对话流程
# 这里定义模型会依次返回的响应
fake_responses = [
    "你好！我是测试助手。",
    "我看到你问的是关于天气的问题。",
    "模拟回答：今天天气晴朗，气温25度。"
]

fake_model = FakeListChatModel(responses=fake_responses, sleep=0.1)

# 加载chat_history.json的辅助函数
def load_chat_history():
    """加载chat_history.json并返回所有sessions"""
    current_dir = Path(__file__).parent
    test_dir = current_dir.parent / "test"
    chat_history_file = test_dir / "chat_history.json"
    
    try:
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
        return chat_history
    except Exception as e:
        print(f"Error loading chat_history.json: {e}")
        return []

def get_current_session(state: AgentState):
    """根据session_index获取当前session数据"""
    chat_history = load_chat_history()
    session_index = state.get("session_index", 0)
    
    if 0 <= session_index < len(chat_history):
        return chat_history[session_index]
    return None

# 4. 定义Agent节点函数
async def call_model(state: AgentState, config: RunnableConfig):
    """调用模拟模型并更新状态"""

    writer = get_stream_writer()
    print(state)
    # 获取最近的用户消息（通常是最后一条）
    last_message = state["messages"][-1]

    print(last_message)

    # 调用模拟模型
    response = fake_model.astream([last_message])
    full_content = ""

    i = 0
    async for chunk in response:
        i += 1
        full_content += chunk.content
        print(chunk)
        print(state['messages'][-1].content)
        yield {
            "messages": [chunk],
        }
        # await copilotkit_emit_state(config, {
        #     "logs": [f"Test State {i}"]
        # })
    yield {
        "num_input": 5,
        "messages": [AIMessage(full_content)],
    }


    # # 将AI响应添加到消息历史
    # return {
    #     "num_input": state["num_input"] + 1 if "num_input" in state else 1,
    #     "messages": [response]
    # }

# async def foo_node(state: AgentState):
#     content = state["messages"][-1].content
#
#     for i in range(10):
#         await asyncio.sleep(0.5)
#         content += f'{i}'
#         yield {"messages": [{"role": "assistant", "content": content}]}

async def print_state(state: AgentState):
    print(state)
    for i in range(10):
        await asyncio.sleep(0.1)
        yield {"num_input": i}
    yield {"num_input": 10}

async def replace_human_message(state: AgentState):
    """替换state.messages中的最后一个条目为当前session的人类输入，并更新指针"""
    # 根据human message数量确定当前session索引
    human_message_count = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage))
    session_index = human_message_count - 1  # 第一个human message对应session 0
    
    # 获取当前session
    chat_history = load_chat_history()
    if 0 <= session_index < len(chat_history):
        session = chat_history[session_index]
        
        # session的第一个元素是user消息
        if len(session) > 0 and session[0].get("role") == "user":
            user_content = session[0].get("content", "")
            
            # 替换最后一条消息为当前session的user消息
            # new_messages = list(state["messages"][:-1]) if state["messages"] else []
            print(state["messages"])
            new_messages= [HumanMessage(content=user_content, id=state["messages"][-1].id)]
            print(new_messages)
            
            # 重置processed_tool_count为0（新session开始），并设置session_index
            yield {
                "messages": new_messages,
                "processed_tool_count": 0,
                "session_index": session_index
            }
        else:
            yield {}
    else:
        # session索引超出范围，直接结束
        yield {}

async def fake_call_model(state: AgentState):
    """检查Agent答复，如果有tool_events则处理，否则逐字回复content"""
    # 获取当前session
    session = get_current_session(state)
    
    if not session or len(session) < 2:
        # 没有assistant消息，直接结束
        yield {}
        return
    
    # session的第二个元素是assistant消息
    assistant_msg = session[1]
    tool_events = assistant_msg.get("tool_events", [])
    processed_tool_count = state.get("processed_tool_count", 0)
    
    # 检查是否还有未处理的tool
    if tool_events and processed_tool_count < len(tool_events):
        # 还有未处理的tool，发送"I'll firstly calling {tool_name}"消息
        current_tool = tool_events[processed_tool_count]
        tool_name = current_tool.get("tool_name", "")

        yield {
            "running_tool_name": tool_name
        }
        return
    
    # 所有tool都已处理，清空running_tool_name并逐字回复content
    content = assistant_msg.get("content", "")
    
    if content:
        # 使用FakeListChatModel逐字回复
        fake_model = FakeListChatModel(responses=[content], sleep=0.005)
        
        # 获取最后一条消息作为输入
        last_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        response = fake_model.astream([last_message])
        
        async for chunk in response:
            yield {
                "messages": [chunk]
            }
        
        # 最后发送完整的AIMessage
        yield {
            "messages": [AIMessage(content=content)],
            "running_tool_name": "",  # 清空running_tool_name
            "processed_tool_count": 0  # 重置tool计数（为下一个session准备）
        }

async def fake_tool(state: AgentState):
    """sleep 8s后返回ToolMessage"""
    # Sleep 8秒
    await asyncio.sleep(8)
    
    # 返回ToolMessage
    tool_message = ToolMessage(content="success", tool_call_id="fake_tool_call")
    
    # 增加已处理的tool计数
    processed_tool_count = state.get("processed_tool_count", 0) + 1
    
    yield {
        "messages": [tool_message],
        "processed_tool_count": processed_tool_count,
        "running_tool_name": ""  # 清空running_tool_name
    }

async def load_fs_state_from_example(state: AgentState):
    """根据当前session的example字段加载对应的example文件并更新fs_state
    只有在确定不会执行fake_tool时才更新状态"""
    # 获取当前session
    session = get_current_session(state)
    
    if not session or len(session) < 2:
        yield {}
        return
    
    # 检查是否会执行fake_tool：如果有未处理的tool_events，则不更新状态
    assistant_msg = session[1]
    tool_events = assistant_msg.get("tool_events", [])
    processed_tool_count = state.get("processed_tool_count", 0)
    
    # 如果还有未处理的tool，则不更新状态
    if tool_events and processed_tool_count < len(tool_events):
        yield {}
        return
    
    # 没有tool或所有tool都已处理，可以更新状态
    # 从assistant消息中获取example编号
    example_num = assistant_msg.get("example", 1)
    
    # NOTE:确定example文件编号（最大为4）
    example_num = min(example_num, 4)
    
    # 构建文件路径（相对于agent目录，test目录在上一级）
    current_dir = Path(__file__).parent
    test_dir = current_dir.parent / "test"
    example_file = test_dir / f"example{example_num}.json"
    
    # 加载JSON文件
    try:
        with open(example_file, 'r', encoding='utf-8') as f:
            example_data = json.load(f)
        
        # 转换数据以匹配FSState结构
        # 处理experiment_id可能是数字的情况
        experiment_id = str(example_data.get("experiment_id", ""))
        
        # 构建fs_state更新
        fs_state_update: FSState = {
            "experiment_id": experiment_id,
            "name": example_data.get("name", ""),
            "description": example_data.get("description", ""),
            "created_date": example_data.get("created_date", ""),
            "tags": example_data.get("tags", []),
            "knowledge_base": example_data.get("knowledge_base", []),
            "status": example_data.get("status", "pending"),
            "start_time": example_data.get("start_time"),
            "end_time": example_data.get("end_time"),
            "duration_seconds": example_data.get("duration_seconds"),
            "research_question": example_data.get("research_question", ""),
            "hypothesis": example_data.get("hypothesis", ""),
            "expected_outcome": example_data.get("expected_outcome", ""),
            "configurations": example_data.get("configurations", []),
            "images": example_data.get("images", []),
            "paths": example_data.get("paths", {
                "experiment_directory": "",
                "manifest_file": "",
                "config_files": {}
            })
        }
        
        yield {
            "fs_state": fs_state_update
        }
        
    except FileNotFoundError:
        print(f"Warning: Example file {example_file} not found")
        yield {}
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from {example_file}: {e}")
        yield {}
    except Exception as e:
        print(f"Error: Failed to load example file {example_file}: {e}")
        yield {}

# 5. 定义判断下一步的逻辑
def should_call_tool(state: AgentState) -> str:
    """判断是否需要调用tool"""
    # 如果running_tool_name不为空，说明需要调用tool
    if state.get("running_tool_name"):
        return "fake_tool"
    
    # 没有tool或所有tool都已处理，结束
    return "end"

# 6. 构建并编译工作流图
def build_agent_graph():
    """构建Agent工作流图"""

    # 创建图构建器
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("replace_human_message", replace_human_message)
    workflow.add_node("fake_call_model", fake_call_model)
    workflow.add_node("fake_tool", fake_tool)
    workflow.add_node("load_fs_state", load_fs_state_from_example)

    # 设置入口点
    workflow.set_entry_point("replace_human_message")

    # replace_human_message -> load_fs_state -> fake_call_model
    workflow.add_edge("replace_human_message", "load_fs_state")
    workflow.add_edge("load_fs_state", "fake_call_model")
    
    # fake_call_model -> 根据是否有tool决定下一步
    workflow.add_conditional_edges(
        "fake_call_model",
        should_call_tool,
        {
            "fake_tool": "fake_tool",
            "end": END
        }
    )
    
    # fake_tool -> fake_call_model (继续处理下一个tool或返回content)
    workflow.add_edge("fake_tool", "load_fs_state")

    # 编译图
    graph = workflow.compile()
    return graph

graph = build_agent_graph()

# # 7. 运行Agent
# if __name__ == "__main__":
#     # 构建Agent
#     agent = build_agent_graph()
#
#     # 准备初始输入
#     initial_state = AgentState(
#         messages=[HumanMessage(content="你好！")],
#         input="你好！"
#     )
#
#     print("=== 开始对话 ===")
#
#     # 运行工作流
#     for step, output in enumerate(agent.stream(initial_state), 1):
#         print(f"\n步骤 {step} 输出:")
#         for message in output.get("messages", []):
#             print(f"  {message.type}: {message.content}")
#
#     print("\n=== 最终状态 ===")
#     final_state = agent.invoke(initial_state)
#     print(f"总消息数: {len(final_state['messages'])}")
#     for i, msg in enumerate(final_state["messages"]):
#         print(f"{i+1}. [{msg.type}] {msg.content[:50]}...")