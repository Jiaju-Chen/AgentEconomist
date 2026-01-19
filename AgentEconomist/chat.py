"""
命令行交互入口

提供命令行交互界面，用于测试 Agent Economist。
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage

from AgentEconomist.graph import build_economist_graph
from AgentEconomist.config import Config


def main():
    """命令行交互模式"""
    print("=" * 60)
    print("Agent Economist (Restructured Version)")
    print("=" * 60)
    print(Config.get_summary())
    print("=" * 60)
    print("Type 'exit' to quit, ':state' to view state, ':history' to view conversation history\n")
    
    agent = build_economist_graph()
    config = {"configurable": {"thread_id": Config.THREAD_ID}}
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        
        if user_input == ":state":
            # Show current state
            snapshot = agent.get_state(config)
            values = getattr(snapshot, "values", {})
            fs_state = values.get("fs_state", {})
            messages = values.get("messages", [])
            print("\n📊 Current State:")
            print(f"  Experiment: {fs_state.get('name', 'N/A')}")
            print(f"  Status: {fs_state.get('status', 'N/A')}")
            print(f"  Messages in state: {len(messages)}")
            continue
        
        if user_input == ":history":
            # Show conversation history
            snapshot = agent.get_state(config)
            values = getattr(snapshot, "values", {})
            messages = values.get("messages", [])
            print(f"\n📜 Conversation History ({len(messages)} messages):")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content = getattr(msg, "content", "")
                if isinstance(content, str) and len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{i}] {msg_type}: {content}")
            continue
        
        if not user_input:
            continue
        
        # Invoke agent
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", "")
                print(f"\n🤖 Agent: {content}")
            else:
                print("\n🤖 Agent: (no response)")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
