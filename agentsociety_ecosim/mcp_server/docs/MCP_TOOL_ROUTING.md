# MCP工具调用和路由机制说明

## 概述

MCP (Model Context Protocol) 使用 JSON-RPC 2.0 协议进行通信。Agent通过发送标准化请求来调用MCP服务器上的工具，MCP服务器根据工具名称路由到相应的处理函数。

## 架构图

```
Agent (Cursor/Claude Desktop)
    ↓ JSON-RPC 2.0 请求
MCP Client
    ↓ STDIO/SSE/HTTP 传输
MCP Server (server_fastmcp.py)
    ↓ 路由匹配
Tool Handler (@mcp.tool() 装饰的函数)
    ↓ 执行
返回结果
```

## 1. MCP如何决定调用哪个tool？

**MCP服务器本身不决定调用哪个tool，而是由Agent决定！**

### Agent的决策过程：

1. **获取工具列表**：Agent首先调用 `tools/list` 获取所有可用工具
   ```json
   {
     "jsonrpc": "2.0",
     "method": "tools/list",
     "id": 1
   }
   ```

2. **接收工具描述**：MCP服务器返回所有已注册的工具列表
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "tools": [
         {
           "name": "get_all_parameters",
           "description": "获取经济仿真系统的所有可配置参数...",
           "inputSchema": {
             "type": "object",
             "properties": {
               "category": {"type": "string", "default": "all"},
               "format": {"type": "string", "default": "json"}
             }
           }
         },
         {
           "name": "analyze_question",
           "description": "分析问题并识别实验类型...",
           ...
         }
       ]
     }
   }
   ```

3. **Agent基于用户请求和工具描述决策**：
   - Agent分析用户的意图
   - 匹配相关工具的描述和参数
   - 决定调用哪个工具

### 示例：
- 用户问："How will AI reshape the labor market?"
- Agent看到 `analyze_question` 工具描述包含"分析问题并识别实验类型"
- Agent决定调用 `analyze_question` 工具

## 2. Agent向MCP发送什么指令？

Agent使用 **JSON-RPC 2.0** 协议发送请求：

### 工具调用请求格式：

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "analyze_question",
    "arguments": {
      "question": "How will AI agents reshape the labor market?"
    },
    "_meta": {
      "progressToken": "optional-token"
    }
  },
  "id": 123
}
```

### 字段说明：

- **`jsonrpc`**: 协议版本，固定为 "2.0"
- **`method`**: 固定为 "tools/call"（调用工具）或 "tools/list"（列出工具）
- **`params.name`**: **工具名称**（这是路由的关键！）
- **`params.arguments`**: 传递给工具的参数（字典格式）
- **`id`**: 请求ID，用于匹配响应

### 其他常用请求：

#### 列出所有工具：
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

#### 获取工具详情（可选）：
```json
{
  "jsonrpc": "2.0",
  "method": "tools/describe",
  "params": {
    "name": "analyze_question"
  },
  "id": 2
}
```

## 3. MCP如何路由找到tool？

### 工具注册机制：

在 `server_fastmcp.py` 中，使用 `@mcp.tool()` 装饰器注册工具：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-economist-parameter-server")

@mcp.tool()
async def analyze_question(question: str) -> str:
    """分析问题并识别实验类型"""
    # 工具实现
    ...
```

### FastMCP的路由机制：

1. **装饰器注册**：
   - `@mcp.tool()` 装饰器会自动提取函数名、参数、文档字符串
   - 将这些信息注册到内部的工具注册表（字典）中
   - 工具名称 = 函数名（如 `analyze_question`）

2. **请求路由过程**：
   ```
   收到 JSON-RPC 请求
       ↓
   解析 method = "tools/call"
       ↓
   提取 params.name = "analyze_question"
       ↓
   在工具注册表中查找 key = "analyze_question"
       ↓
   找到对应的函数对象
       ↓
   验证参数类型（基于函数签名和类型注解）
       ↓
   调用函数：analyze_question(**arguments)
       ↓
   返回结果（JSON字符串）
   ```

3. **工具注册表示例**（概念性的）：
   ```python
   tools_registry = {
       "analyze_question": {
           "function": <function analyze_question at 0x...>,
           "name": "analyze_question",
           "description": "分析问题并识别实验类型...",
           "parameters": {
               "question": {"type": "string", "required": True}
           }
       },
       "get_all_parameters": {
           "function": <function get_all_parameters at 0x...>,
           ...
       },
       ...
   }
   ```

### 路由匹配过程（伪代码）：

```python
# FastMCP内部实现（简化版）
class FastMCP:
    def __init__(self):
        self.tools = {}  # 工具注册表
    
    def tool(self):
        """装饰器：注册工具"""
        def decorator(func):
            self.tools[func.__name__] = {
                "function": func,
                "name": func.__name__,
                "description": func.__doc__,
                "parameters": extract_parameters(func)
            }
            return func
        return decorator
    
    async def handle_request(self, request):
        """处理JSON-RPC请求"""
        if request["method"] == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"]["arguments"]
            
            # 路由：根据工具名称查找
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = await tool["function"](**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"content": [{"type": "text", "text": result}]}
                }
            else:
                # 工具不存在
                return {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32601, "message": "Method not found"}
                }
```

## 实际示例

### 完整的调用流程：

1. **Agent获取工具列表**：
   ```json
   // Agent发送
   {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
   
   // MCP返回（部分）
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "tools": [
         {
           "name": "analyze_question",
           "description": "分析问题并识别实验类型..."
         }
       ]
     }
   }
   ```

2. **Agent调用工具**：
   ```json
   // Agent发送
   {
     "jsonrpc": "2.0",
     "method": "tools/call",
     "params": {
       "name": "analyze_question",
       "arguments": {
         "question": "How will AI reshape labor market?"
       }
     },
     "id": 2
   }
   ```

3. **MCP路由和执行**：
   - FastMCP收到请求
   - 提取 `params.name = "analyze_question"`
   - 在注册表中查找 `mcp.tools["analyze_question"]`
   - 找到对应的函数：`analyze_question()`
   - 调用：`await analyze_question(question="How will AI reshape labor market?")`
   - 返回结果

4. **MCP返回结果**：
   ```json
   {
     "jsonrpc": "2.0",
     "id": 2,
     "result": {
       "content": [
         {
           "type": "text",
           "text": "{\"question_type\": \"labor_productivity\", ...}"
         }
       ]
     }
   }
   ```

## 关键点总结

1. **工具注册**：使用 `@mcp.tool()` 装饰器，工具名称 = 函数名
2. **路由机制**：基于 `params.name` 字段，在工具注册表中查找
3. **Agent决策**：Agent根据工具描述和用户意图决定调用哪个工具
4. **协议**：JSON-RPC 2.0，通过 STDIO/SSE/HTTP 传输

## 调试工具调用

如果要调试工具调用，可以：

1. **查看工具列表**：
   ```bash
   # 启动MCP服务器时会打印所有注册的工具
   python server_fastmcp.py
   ```

2. **查看工具注册**：
   ```python
   # 在server_fastmcp.py中添加
   print(f"Registered tools: {list(mcp.tools.keys())}")
   ```

3. **日志记录**：
   FastMCP会自动记录工具调用的日志（如果启用）

## 常见问题

### Q: 如何添加新工具？
A: 使用 `@mcp.tool()` 装饰器装饰函数，FastMCP会自动注册。

### Q: 工具名称冲突怎么办？
A: 工具名称 = 函数名，确保函数名唯一即可。

### Q: Agent如何知道调用哪个工具？
A: Agent根据工具的描述（docstring）和参数要求，结合用户意图进行匹配。

### Q: 可以动态添加工具吗？
A: FastMCP支持动态添加，但需要重启服务器或使用特殊的注册机制。

