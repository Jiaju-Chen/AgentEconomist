# test_workflow.py
from langgraph_sdk import get_sync_client
import time

client = get_sync_client(url="http://localhost:8123")

THREAD_ID = "test-thread-123"


def test_full_workflow():
    """Test complete workflow interaction"""

    # 1. CREATE NEW THREAD
    print("🧵 Creating new thread...")
    thread = client.threads.create()
    print(f"Thread ID: {thread["thread_id"]}")

    # 2. START WORKFLOW (streaming)
    print("\n🚀 Starting agent...")
    stream = client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="sample_agent",  # From langgraph.json
        input={
            "messages": [{"role": "user", "content": "What's the weather in SF?"}]
        },
        stream_mode=["messages", "updates"]
    )

    for chunk in stream:
        print(f"📡 {chunk.event}: {chunk.data}")

    # 3. INSPECT STATE
    print("\n🔍 Current thread state:")
    state = client.threads.get_state(thread["thread_id"])
    print(f"State: {state}")

    # # 4. CONTINUE CONVERSATION
    # print("\n💬 Sending follow-up...")
    # follow_up = client.runs.stream(
    #     thread_id=thread.thread_id,
    #     assistant_id="agent",
    #     input={
    #         "messages": [{"role": "user", "content": "What about NYC?"}]
    #     },
    #     stream_mode="updates"
    # )
    #
    # for chunk in follow_up:
    #     print(f"📡 {chunk.event}: {chunk.data}")
    #
    # # 5. LIST ALL THREADS
    # print("\n📋 All threads:")
    # threads = client.threads.search()
    # for t in threads:
    #     print(f"  - {t.thread_id}")


if __name__ == '__main__':

    test_full_workflow()