"""
LangGraph backend entrypoint for Agent Economist.

目标（最小改动版）：
- 不改现有工具的业务逻辑（复用 economist/design_agent.py 的工具函数）
- 仅用 LangGraph 来管理 LLM <-> tools 的调度
- 每次 tool call 后，基于 manifest.yaml 调用 agent_state.manifest_to_agent_state，
  并将结果写入与 manifest 同目录下的 agent_state.json（供前端/调试读取）
"""

from __future__ import annotations

import json
import os
import re
import inspect
import functools
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _ensure_repo_on_syspath() -> None:
    """
    Ensure this repo's root directory is importable.

    Why:
    - Running `python economist/langgraph_agent.py` does NOT automatically add repo root to sys.path.
    - We need repo root for `import agent_state` (agent_state.py lives at repo root).
    - Keep this minimal and deterministic (no env required).
    """
    economist_dir = Path(__file__).resolve().parent
    repo_root = economist_dir.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _require_langgraph_deps() -> None:
    """Fail fast with a clear message if deps are missing."""
    try:
        import langgraph  # noqa: F401
        import langchain  # noqa: F401
        import langchain_openai  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing LangGraph dependencies. Please install via poetry:\n"
            "  poetry install\n"
            "or ensure these deps exist:\n"
            "  langgraph, langchain, langchain-openai\n"
            f"Original error: {e}"
        )


def _extract_first_tag(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip() or None


def _tool_response_to_text_flexible(response: Any) -> str:
    """
    Best-effort conversion of AgentScope ToolResponse (and similar) into plain text.

    Why:
    - AgentScope ToolResponse.content blocks may expose `.text`, `.content`, or be dict-like,
      depending on version.
    - If we fail to extract text, LangGraph tool output becomes empty, and the model keeps retrying.
    """
    if response is None:
        return ""

    content = getattr(response, "content", None)
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            text = None
            if hasattr(block, "text"):
                text = getattr(block, "text", None)
            if not text and hasattr(block, "content"):
                text = getattr(block, "content", None)
            if not text and isinstance(block, dict):
                text = block.get("text") or block.get("content")
            if text:
                parts.append(str(text))
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    # Fallback
    s = str(response).strip()
    return s


def _write_agent_state_json(manifest_path: str) -> Optional[str]:
    """
    从 manifest.yaml 生成 agent_state dict，并写入 agent_state.json（同目录）。
    返回写入文件路径（失败返回 None）。
    """
    try:
        _ensure_repo_on_syspath()
        from agent_state import manifest_to_agent_state
    except Exception as e:
        raise RuntimeError(
            "Failed to import agent_state.manifest_to_agent_state. "
            "Ensure repo root is on PYTHONPATH and dependencies are installed."
        ) from e

    try:
        manifest_abs = os.path.abspath(os.path.expanduser(manifest_path))
        state = manifest_to_agent_state(manifest_abs)
        out_path = os.path.join(os.path.dirname(manifest_abs), "agent_state.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate/write agent_state.json from manifest: {manifest_path}"
        ) from e


def _wrap_tool_and_update_state(fn, *, manifest_kw: str = "manifest_path"):
    """
    Wrap a tool function:
    - call original tool function (business logic unchanged)
    - try to find manifest path (prefer kwargs[manifest_kw], else parse output tags)
    - return plain text (ToolResponse -> text)

    NOTE:
    - To keep behavior identical to economist/design_agent.py, we DO NOT append extra tags
      to the tool output (e.g. <agent_state_json>...</agent_state_json>).
    - The agent_state.json update is handled as a side-effect in the graph node `run_tools`,
      after each tool call, without mutating tool outputs.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Some LangChain/LangGraph tool runners may wrap arguments as {"args": [...], "kwargs": {...}}
        # when the function signature is not fully introspectable. Unwrap for compatibility.
        if "args" in kwargs or "kwargs" in kwargs:
            packed_args = kwargs.pop("args", None)
            packed_kwargs = kwargs.pop("kwargs", None)
            if packed_kwargs:
                if not isinstance(packed_kwargs, dict):
                    raise TypeError(f"Tool runner passed non-dict kwargs: {type(packed_kwargs).__name__}")
                kwargs.update(packed_kwargs)
            if packed_args is not None:
                if isinstance(packed_args, (list, tuple)):
                    args = tuple(packed_args)
                elif isinstance(packed_args, dict):
                    # Sometimes args is actually a dict of named arguments
                    kwargs.update(packed_args)
                    args = ()
                else:
                    raise TypeError(f"Tool runner passed non-list/dict args: {type(packed_args).__name__}")

        resp = fn(*args, **kwargs)
        text = _tool_response_to_text_flexible(resp)
        if not text:
            text = "<status>error</status><error>Empty tool output (conversion failed)</error>"

        return text

    # Preserve name/doc for tool schema readability (even if signature differs).
    wrapper.__name__ = getattr(fn, "__name__", "wrapped_tool")
    wrapper.__doc__ = getattr(fn, "__doc__", "") or ""
    # Preserve signature so tool schema matches the real tool (prevents model from sending args/kwargs wrapper objects)
    try:
        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Preserve annotations for Pydantic schema generation (avoid KeyError on missing type hints)
    try:
        wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {})
    except Exception:
        pass
    return wrapper


def build_langgraph_workflow():
    """Build an explicit StateGraph workflow like test_mcp.py.

    Graph:
        START -> init_node -> call_model
        call_model --(tools_condition)--> tools -> sync_agent_state -> call_model
        call_model --(no tool_calls)--> END
    """
    _require_langgraph_deps()
    _ensure_repo_on_syspath()

    from langchain_core.messages import SystemMessage
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.constants import END, START
    from langgraph.graph import MessagesState, StateGraph
    from langgraph.prebuilt import tools_condition

    import design_agent as da

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or DASHSCOPE_API_KEY) environment variable.")

    # LLM config: keep it env-driven to avoid hardcoding.
    model_name = os.getenv("ECONOMIST_LLM_MODEL", "gpt-5")
    base_url = os.getenv("ECONOMIST_LLM_BASE_URL") or os.getenv("BASE_URL") or None

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("ECONOMIST_LLM_TEMPERATURE", "0.1")),
    )

    # Reuse the existing system prompt (tool docs + parameter guide + workflow rules).
    system_prompt = da._build_sys_prompt()

    # Wrap tools (business logic unchanged; only add agent_state update as a tail step).
    tools = [
        StructuredTool.from_function(_wrap_tool_and_update_state(da.get_available_parameters), name="get_available_parameters"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.get_parameter_info), name="get_parameter_info"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.init_experiment_manifest), name="init_experiment_manifest"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.create_yaml_from_template), name="create_yaml_from_template"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.run_simulation), name="run_simulation"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.read_simulation_report), name="read_simulation_report"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.compare_experiments), name="compare_experiments"),
        StructuredTool.from_function(_wrap_tool_and_update_state(da.analyze_experiment_directory), name="analyze_experiment_directory"),
    ]

    # Knowledge base tools are optional (import may fail in some environments).
    try:
        import sys

        # database/ sits next to economist/ under repo root
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../economist
        database_path = os.path.abspath(os.path.join(current_dir, "../database"))
        if database_path not in sys.path:
            sys.path.insert(0, database_path)

        from knowledge_base.tool import (
            query_knowledge_base as _query_kb,
            find_similar_papers as _find_similar,
            get_paper_details as _get_paper_details,
        )

        def query_knowledge_base(
            query: str,
            top_k: int = 20,
            journals: str = "",
            year_start: int = 0,
            year_end: int = 0,
            doc_type: str = "",
        ) -> str:
            """
            查询学术论文知识库（与 design_agent.py 行为对齐）。

            CRITICAL: query 必须是英文。若包含中文，知识库工具会返回 error。
            """
            result = _query_kb(
                query=query,
                top_k=top_k,
                journals=journals if journals else None,
                year_start=year_start if year_start > 0 else None,
                year_end=year_end if year_end > 0 else None,
                doc_type=doc_type or None,
            )
            if result.get("status") == "success":
                results_json = json.dumps(result.get("results", []), ensure_ascii=False, indent=2)
                return (
                    f"<status>success</status>"
                    f"<query>{query}</query>"
                    f"<total_found>{result.get('total_found', 0)}</total_found>"
                    f"<results>{results_json}</results>"
                )
            return f"<status>error</status><error>{result.get('error', 'Unknown error')}</error>"

        def find_similar_papers(
            title: str,
            abstract: str = "",
            top_k: int = 5,
        ) -> str:
            result = _find_similar(title=title, abstract=abstract, top_k=top_k)
            if result.get("status") == "success":
                results_json = json.dumps(result.get("similar_papers", []), ensure_ascii=False, indent=2)
                return (
                    f"<status>success</status>"
                    f"<query_title>{title}</query_title>"
                    f"<total_found>{result.get('total_found', 0)}</total_found>"
                    f"<similar_papers>{results_json}</similar_papers>"
                )
            return f"<status>error</status><error>{result.get('error', 'Unknown error')}</error>"

        def get_paper_details(
            paper_id: str,
            include_similar: bool = True,
            similar_count: int = 3,
        ) -> str:
            # include_similar/similar_count are kept for signature parity; underlying tool may ignore.
            result = _get_paper_details(paper_id=paper_id)
            if result.get("status") == "success":
                return json.dumps(result, ensure_ascii=False, indent=2)
            return f"<status>error</status><error>{result.get('error', 'Unknown error')}</error>"

        tools.extend(
            [
                StructuredTool.from_function(
                    _wrap_tool_and_update_state(query_knowledge_base, manifest_kw="__none__"),
                    name="query_knowledge_base",
                ),
                StructuredTool.from_function(
                    _wrap_tool_and_update_state(find_similar_papers, manifest_kw="__none__"),
                    name="find_similar_papers",
                ),
                StructuredTool.from_function(
                    _wrap_tool_and_update_state(get_paper_details, manifest_kw="__none__"),
                    name="get_paper_details",
                ),
            ]
        )
    except Exception as e:
        print("warning: Failed to set up knowledge base tools. ", e)
        # raise RuntimeError(
        #     "Failed to set up knowledge base tools. "
        #     "If you don't need KB tools during debugging, comment out the KB tools block in "
        #     "economist/langgraph_agent.py."
        # ) from e

    llm_with_tools = llm.bind_tools(tools)

    class EconomistState(MessagesState):
        # 参照 test_mcp.py：在 MessagesState 上扩展业务状态字段
        inited: bool
        # 最近一次从 manifest 派生的 agent_state（结构见 AGENT_STATE_SPECIFICATION.md）
        agent_state: dict | None
        # 最近一次工具输出提到的 manifest path
        manifest_path: str | None
        # 最近一次落盘的 agent_state.json
        agent_state_json: str | None

    def init_node(state: EconomistState):
        if state.get("inited", False):
            return {}
        return {
            "messages": [SystemMessage(content=system_prompt)],
            "inited": True,
            "agent_state": state.get("agent_state"),
            "manifest_path": state.get("manifest_path"),
            "agent_state_json": state.get("agent_state_json"),
        }

    def call_model(state: EconomistState):
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}

    tools_by_name = {t.name: t for t in tools}

    def run_tools(state: EconomistState):
        """
        Execute tool calls from the latest AIMessage.

        Why custom (instead of ToolNode):
        - Some LC/LG versions swallow tool exceptions and produce empty ToolMessage.content.
        - For debugging we want: tool output always visible; exceptions raise with traceback.
        """
        from langchain_core.messages import ToolMessage

        msgs = state.get("messages", [])
        if not msgs:
            return {}
        last = msgs[-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        out_messages = []
        last_manifest_path: str | None = None
        last_agent_state_json: str | None = None
        last_agent_state: dict | None = None

        for tc in tool_calls:
            name = tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id") or ""
            args_obj = tc.get("args", {}) or {}

            if not name or name not in tools_by_name:
                raise RuntimeError(f"Unknown tool requested by model: {name!r}")

            tool = tools_by_name[name]
            # StructuredTool.invoke expects a dict of args.
            if isinstance(args_obj, dict):
                result_text = tool.invoke(args_obj)
            else:
                # Extremely defensive: if args isn't a dict, pass as single positional via wrapper path
                result_text = tool.invoke({"args": args_obj})

            if result_text is None:
                result_text = ""
            if not isinstance(result_text, str):
                result_text = str(result_text)

            # Side-effect: after EACH tool call, if we can locate manifest_path, update agent_state.json.
            manifest_path = None
            if isinstance(args_obj, dict):
                manifest_path = args_obj.get("manifest_path")
            if not manifest_path:
                manifest_path = _extract_first_tag(result_text, "manifest_updated") or _extract_first_tag(
                    result_text, "manifest_path"
                )
            if manifest_path:
                last_manifest_path = manifest_path
                last_agent_state_json = _write_agent_state_json(manifest_path)
                try:
                    with open(last_agent_state_json, "r", encoding="utf-8") as f:
                        last_agent_state = json.load(f)
                except Exception:
                    # keep fail-fast elsewhere; reading this is optional for state
                    last_agent_state = None

            # ToolMessage signature differs across versions; handle both.
            try:
                out_messages.append(ToolMessage(content=result_text, name=name, tool_call_id=tc_id))
            except TypeError:
                tm = ToolMessage(content=result_text, tool_call_id=tc_id)
                # best-effort attach name for downstream debugging/trace
                try:
                    tm.name = name  # type: ignore[attr-defined]
                except Exception:
                    pass
                out_messages.append(tm)

        updates: dict[str, Any] = {"messages": out_messages}
        if last_manifest_path:
            updates["manifest_path"] = last_manifest_path
        if last_agent_state_json:
            updates["agent_state_json"] = last_agent_state_json
        if last_agent_state is not None:
            updates["agent_state"] = last_agent_state
        return updates

    def sync_agent_state(state: EconomistState):
        """在 tools 执行后，把工具返回中附带的 manifest/agent_state_json 同步进 Graph State。"""
        msgs = state.get("messages", [])
        manifest_path = None
        agent_state_json = None

        # 从最后一条 tool message 里解析（ToolNode 会把 tool 输出写入 messages）
        for m in reversed(msgs):
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                continue
            if manifest_path is None:
                manifest_path = _extract_first_tag(content, "manifest_path") or _extract_first_tag(
                    content, "manifest_updated"
                )
            if agent_state_json is None:
                agent_state_json = _extract_first_tag(content, "agent_state_json")
            if manifest_path or agent_state_json:
                break

        new_state: dict[str, Any] = {}
        if manifest_path:
            new_state["manifest_path"] = manifest_path
        if agent_state_json:
            new_state["agent_state_json"] = agent_state_json
            try:
                with open(agent_state_json, "r", encoding="utf-8") as f:
                    new_state["agent_state"] = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read/parse agent_state.json: {agent_state_json}"
                ) from e
        return new_state

    workflow = StateGraph(EconomistState)
    workflow.add_node("init_node", init_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", run_tools)
    workflow.add_node("sync_agent_state", sync_agent_state)

    workflow.add_edge(START, "init_node")
    workflow.add_edge("init_node", "call_model")
    workflow.add_edge("tools", "sync_agent_state")
    workflow.add_edge("sync_agent_state", "call_model")

    workflow.add_conditional_edges(
        "call_model",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    return workflow.compile(checkpointer=MemorySaver())


def _get_messages_from_graph_state(agent, config: dict) -> list:
    """
    从 LangGraph checkpointer 中读取当前 thread 的完整 messages（全流程对话）。
    兼容不同版本的 langgraph StateSnapshot 结构。
    """
    snap = agent.get_state(config)
    values = getattr(snap, "values", None)
    if isinstance(values, dict):
        return values.get("messages", []) or []
    # fallback: some versions may return a dict directly
    if isinstance(snap, dict):
        return snap.get("messages", []) or []
    return []


def _format_messages_for_print(messages: list) -> str:
    """把 messages 格式化为可读的对话文本（用于调试输出）。"""
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except Exception as e:
        raise RuntimeError("Failed to import langchain_core.messages for formatting.") from e

    lines: list[str] = []
    for m in messages:
        role = "message"
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = f"tool:{getattr(m, 'name', '') or getattr(m, 'tool', '') or 'unknown'}"

        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"{role}> {content}")
    return "\n".join(lines)


def _default_trace_path(thread_id: str) -> str:
    base_dir = Path(__file__).resolve().parent / "traces"
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_thread = re.sub(r"[^a-zA-Z0-9_.-]+", "_", thread_id or "chat")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(base_dir / f"trace_{safe_thread}_{ts}.jsonl")


def _truncate_text(s: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 12)] + "...(truncated)"


def _message_to_dict(m: Any, *, truncate: int, include_system: bool) -> dict:
    """
    Convert LangChain message objects to JSON-serializable dict.
    Captures tool_calls (AIMessage) and tool name/content (ToolMessage).
    """
    msg_type = m.__class__.__name__
    content = getattr(m, "content", "")
    if not isinstance(content, str):
        content = str(content)

    # Avoid huge logs by default; system prompt can be massive.
    if msg_type == "SystemMessage" and not include_system:
        content = "(omitted system prompt)"
    content = _truncate_text(content, truncate)

    d: dict[str, Any] = {
        "type": msg_type,
        "content": content,
    }

    # Tool calls live on AIMessage in LC; keep raw structure for debugging
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = tool_calls

    # ToolMessage name differs by version; keep both if present
    name = getattr(m, "name", None)
    tool = getattr(m, "tool", None)
    if name:
        d["name"] = name
    if tool:
        d["tool"] = tool

    # Preserve metadata that may contain tool execution errors depending on LC/LG versions
    additional_kwargs = getattr(m, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        d["additional_kwargs"] = additional_kwargs
    response_metadata = getattr(m, "response_metadata", None)
    if isinstance(response_metadata, dict) and response_metadata:
        d["response_metadata"] = response_metadata
    # Some message classes carry an artifact (e.g. tool error details)
    artifact = getattr(m, "artifact", None)
    if artifact is not None:
        try:
            json.dumps(artifact, ensure_ascii=False)
            d["artifact"] = artifact
        except Exception:
            d["artifact"] = str(artifact)

    return d


def _append_trace(trace_path: str, event: dict) -> None:
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def main():
    """
    Minimal CLI loop for LangGraph agent.
    """
    agent = build_langgraph_workflow()

    print("✅ LangGraph Agent Economist started.")
    print("Type 'exit' to quit.\n")

    from langchain_core.messages import HumanMessage

    thread_id = os.getenv("ECONOMIST_THREAD_ID", "economist-chat-1")
    config = {"configurable": {"thread_id": thread_id}}

    # Full trace log (JSONL) for debugging/replay: dialog + toolcalls + state snapshot
    trace_path = os.getenv("ECONOMIST_TRACE_PATH") or _default_trace_path(thread_id)
    truncate = int(os.getenv("ECONOMIST_TRACE_TRUNCATE", "8000"))
    include_system = os.getenv("ECONOMIST_TRACE_INCLUDE_SYSTEM", "0").lower() in {"1", "true", "yes", "y"}

    _append_trace(
        trace_path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "session_start",
            "thread_id": thread_id,
            "trace_path": trace_path,
            "truncate": truncate,
            "include_system": include_system,
        },
    )
    print(f"trace> {trace_path}\n")

    while True:
        user_input = input("user> ").strip()
        if user_input.lower() == "exit":
            _append_trace(
                trace_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "session_end",
                    "thread_id": thread_id,
                },
            )
            break

        # Debug helpers (read dialog/state directly from LangGraph)
        if user_input in {":dialog", ":history"}:
            msgs = _get_messages_from_graph_state(agent, config)
            print(_format_messages_for_print(msgs) + "\n")
            continue
        if user_input == ":state":
            snap = agent.get_state(config)
            values = getattr(snap, "values", None)
            if isinstance(values, dict):
                print(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
            else:
                print(str(snap) + "\n")
            continue

        _append_trace(
            trace_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "user_input",
                "thread_id": thread_id,
                "text": user_input,
            },
        )

        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        # result is typically a dict with messages
        msgs = result.get("messages", [])
        if msgs:
            print(f"assistant> {msgs[-1].content}\n")
        else:
            print("assistant> (no response)\n")

        # After each turn, dump full dialog + full state snapshot (for replay/diagnosis)
        snap = agent.get_state(config)
        values = getattr(snap, "values", None)
        if not isinstance(values, dict):
            values = {}
        full_msgs = values.get("messages", []) or []

        _append_trace(
            trace_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "turn_end",
                "thread_id": thread_id,
                "assistant_last": _truncate_text(getattr(msgs[-1], "content", "") if msgs else "", truncate),
                "messages": [
                    _message_to_dict(m, truncate=truncate, include_system=include_system) for m in full_msgs
                ],
                # state snapshot (exclude full messages to avoid duplication)
                "state": {k: v for k, v in values.items() if k != "messages"},
            },
        )


if __name__ == "__main__":
    main()


