#!/usr/bin/env python3
"""
LangGraph Server 启动脚本
使用 FastAPI 启动 LangGraph Agent 服务（Python 3.11+）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langgraph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

from AgentEconomist.graph.agent import build_economist_graph
from AgentEconomist.config import Config

# 创建 FastAPI 应用
app = FastAPI(title="Agent Economist API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 编译 graph
checkpointer = MemorySaver()
graph: CompiledGraph = build_economist_graph()

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Agent Economist",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# LangGraph API 兼容端点
@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: dict):
    """创建并运行一个 thread"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # 从请求中提取消息
    messages = request.get("input", {}).get("messages", [])
    if not messages:
        return {"error": "No messages provided"}
    
    # 调用 graph
    result = graph.invoke({"messages": messages}, config=config)
    
    return {
        "run_id": f"run_{thread_id}",
        "status": "success",
        "result": result
    }

@app.get("/threads/{thread_id}/state")
async def get_state(thread_id: str):
    """获取 thread 的状态"""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    return {
        "thread_id": thread_id,
        "state": {
            "values": state.values if hasattr(state, "values") else {},
            "next": state.next if hasattr(state, "next") else []
        }
    }

@app.post("/threads/{thread_id}/stream")
async def stream_run(thread_id: str, request: dict):
    """流式运行（简化版，返回完整结果）"""
    config = {"configurable": {"thread_id": thread_id}}
    messages = request.get("input", {}).get("messages", [])
    
    if not messages:
        return {"error": "No messages provided"}
    
    # 流式调用
    events = []
    async for event in graph.astream({"messages": messages}, config=config):
        events.append(event)
    
    return {
        "events": events,
        "thread_id": thread_id
    }

if __name__ == "__main__":
    Config.validate()
    
    print("=" * 60)
    print("🚀 Starting Agent Economist Server")
    print("=" * 60)
    print(f"📍 Project Root: {Config.PROJECT_ROOT}")
    print(f"🌐 Server: {Config.LANGGRAPH_HOST}:{Config.LANGGRAPH_PORT}")
    print(f"🤖 Model: {Config.LLM_MODEL}")
    print("=" * 60)
    print("\n✅ Server ready! Press Ctrl+C to stop.\n")
    
    uvicorn.run(
        app,
        host=Config.LANGGRAPH_HOST,
        port=Config.LANGGRAPH_PORT,
        log_level="info"
    )
