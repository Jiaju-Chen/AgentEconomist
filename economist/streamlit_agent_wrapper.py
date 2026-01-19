"""
Streamlit Agent Wrapper
用于在 Streamlit 中运行 Design Agent 的包装器，支持流式输出
"""

import asyncio
import queue
import threading
from typing import Optional, Callable, Generator
from agentscope.message import Msg


class StreamlitAgentWrapper:
    """将 AgentScope agent 包装为可在 Streamlit 中使用的形式，支持流式输出"""
    
    def __init__(self, agent, user_agent):
        self.agent = agent
        self.user_agent = user_agent
        self.message_queue = queue.Queue()
        self.is_running = False
        self._stream_buffer = queue.Queue()
        self._stream_complete = threading.Event()
        self._last_printed_content = ""  # 跟踪已打印的内容，避免重复
        self._latest_tool_events: list[dict] = []
        
        # 保存原始的 print 方法
        self._original_print = self.agent.print if hasattr(self.agent, 'print') else None
        
        # 如果 agent 有 print 方法，替换它以捕获流式输出
        if self._original_print:
            # 使用闭包捕获 self 的引用
            wrapper_self = self
            async def intercepted_print(msg, flush=True):
                """拦截 print 调用，将内容放入缓冲区，实现实时流式输出
                
                在 ReActAgent 的流式模式下，_reasoning 阶段会通过 
                await self.print(msg, False) 实时打印内容。
                我们拦截这个调用，提取内容并放入缓冲区，这样就能在工具调用前
                实时显示 reasoning 输出。
                """
                # 调用原始 print（保持原有行为，例如打印到控制台）
                if wrapper_self._original_print:
                    await wrapper_self._original_print(msg, flush)
                
                # 提取内容并放入缓冲区
                if msg and hasattr(msg, 'content'):
                    try:
                        content = msg.get_text_content() if hasattr(msg, 'get_text_content') else str(msg.content)
                        if content and isinstance(content, str):
                            # 计算增量内容（新内容 - 已打印内容）
                            # 在流式模式下，print 会被多次调用，每次 content 都包含累积内容
                            last = wrapper_self._last_printed_content
                            
                            # 如果内容完全相同，跳过（避免重复）
                            if content == last:
                                return
                            
                            # 如果新内容以已打印内容开头，提取增量
                            if last and content.startswith(last):
                                new_chunk = content[len(last):]
                                if new_chunk:  # 即使只有空白也要处理（保留格式）
                                    wrapper_self._stream_buffer.put(new_chunk)
                                    wrapper_self._last_printed_content = content
                            elif not last:
                                # 第一次打印，直接使用
                                if content:
                                    wrapper_self._stream_buffer.put(content)
                                    wrapper_self._last_printed_content = content
                            else:
                                # 内容结构可能发生变化（这种情况应该很少，但在某些情况下可能发生）
                                # 尝试找最长公共前缀
                                common_prefix_len = 0
                                min_len = min(len(last), len(content))
                                for i in range(min_len):
                                    if last[i] == content[i]:
                                        common_prefix_len = i + 1
                                    else:
                                        break
                                
                                if common_prefix_len > 0:
                                    # 有公共前缀，提取增量
                                    new_chunk = content[common_prefix_len:]
                                    if new_chunk:
                                        wrapper_self._stream_buffer.put(new_chunk)
                                        wrapper_self._last_printed_content = content
                                elif len(content) > len(last):
                                    # 内容完全不同但更长，可能是完全重写
                                    # 为避免重复，只输出新增部分（但这可能不准确）
                                    # 在这种情况下，我们选择不输出，避免混乱
                                    wrapper_self._last_printed_content = content
                    except Exception as e:
                        # 如果提取内容出错，不影响原始 print 调用
                        pass  # 静默失败，避免干扰原始输出
            
            # 替换 agent 的 print 方法
            self.agent.print = intercepted_print
    
    def _stream_response_generator(self) -> Generator[str, None, None]:
        """生成器函数，用于流式输出响应
        
        Streamlit 的 st.write_stream 期望接收增量内容，它会自动累积并显示。
        我们只 yield 新的增量 chunk，Streamlit 会自动处理累积显示。
        
        注意：为了避免重复，我们确保每个 chunk 只 yield 一次。
        """
        accumulated = ""  # 本地累积，用于确保不重复
        
        while True:
            try:
                # 从缓冲区获取内容块，设置超时避免无限等待
                chunk = self._stream_buffer.get(timeout=0.1)
                
                if chunk is None:  # 结束标记
                    break
                
                # 累积内容
                accumulated += chunk
                # 只 yield 增量 chunk（Streamlit 会自动处理累积显示）
                yield chunk
                    
            except queue.Empty:
                # 检查是否完成
                if self._stream_complete.is_set():
                    break
                # 如果还没完成，等待下一个chunk
                continue
    
    async def _process_message_async(self, user_input: str):
        """异步处理消息，通过拦截 print 方法实时捕获输出"""
        try:
            user_msg = Msg(name="user", role="user", content=user_input)
            
            # 调用 agent，此时 print 方法已被拦截，会实时将内容放入缓冲区
            response_msg = await self.agent(user_msg)
            
            # 获取最终完整内容（用于降级处理和保存）
            content = response_msg.get_text_content() if hasattr(response_msg, 'get_text_content') else str(response_msg)
            
            # 确保所有内容都已放入缓冲区
            # 如果缓冲区为空但内容存在，说明 print 拦截可能没有工作，需要降级处理
            if content:
                # 检查是否有内容没有被捕获（通过比较最后打印的内容）
                if self._last_printed_content != content:
                    # 有一些内容没有被捕获，可能是最后的增量
                    if content.startswith(self._last_printed_content):
                        remaining = content[len(self._last_printed_content):]
                        if remaining:
                            self._stream_buffer.put(remaining)
                            self._last_printed_content = content
                    elif not self._last_printed_content:
                        # 如果完全没有被捕获，可能是 print 拦截没有工作
                        # 降级处理：手动分段输出（但这种情况应该很少）
                        if self._stream_buffer.empty():
                            paragraphs = content.split('\n\n')
                            for para_idx, paragraph in enumerate(paragraphs):
                                if paragraph.strip():
                                    self._stream_buffer.put(paragraph)
                                    await asyncio.sleep(0.01)
                                if para_idx < len(paragraphs) - 1:
                                    self._stream_buffer.put('\n\n')
                                    await asyncio.sleep(0.01)
                            self._last_printed_content = content
            
            # 标记完成
            self._stream_buffer.put(None)  # 结束标记
            self._stream_complete.set()
            logger = getattr(self.agent, "tool_call_logger", None)
            if logger:
                self._latest_tool_events = logger.drain()
            else:
                self._latest_tool_events = []
            
            return content
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._stream_buffer.put(error_msg)
            self._stream_buffer.put(None)
            self._stream_complete.set()
            import traceback
            traceback.print_exc()
            return error_msg
    
    async def process_message(
        self,
        user_input: str,
        on_update: Optional[Callable] = None,
    ):
        """处理用户消息并返回 agent 响应（兼容旧接口）"""
        # 重置流式输出状态
        self._stream_buffer = queue.Queue()
        self._stream_complete = threading.Event()
        self._last_printed_content = ""  # 重置已打印内容跟踪
        
        # 启动异步处理
        task = asyncio.create_task(self._process_message_async(user_input))
        
        # 等待完成
        content = await task
        return content
    
    def process_sync_streaming(
        self,
        user_input: str,
        on_update: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """同步处理消息并支持流式输出（用于 Streamlit st.write_stream）
        
        关键：通过拦截 agent 的 print 方法，可以实时捕获工具调用前的 reasoning 输出
        """
        # 重置流式输出状态
        self._stream_buffer = queue.Queue()
        self._stream_complete = threading.Event()
        self._last_printed_content = ""  # 重置已打印内容跟踪
        
        # 在新线程中运行异步处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        def run_async():
            try:
                loop.run_until_complete(self._process_message_async(user_input))
            finally:
                loop.close()
        
        # 启动后台线程（agent 处理会在后台进行，print 拦截会将内容放入缓冲区）
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        
        # 返回生成器（实时从缓冲区读取并输出）
        return self._stream_response_generator()
    
    def process_sync(self, user_input: str):
        """同步处理消息（用于 Streamlit，向后兼容）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_message(user_input))
        finally:
            loop.close()

    def pop_latest_tool_events(self) -> list[dict]:
        events = getattr(self, "_latest_tool_events", [])
        self._latest_tool_events = []
        return events


def create_agent_for_streamlit():
    """创建用于 Streamlit 的 agent 实例"""
    import os
    from design_agent import build_design_agent

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量")

    agent, user = build_design_agent(api_key)
    return StreamlitAgentWrapper(agent, user)

