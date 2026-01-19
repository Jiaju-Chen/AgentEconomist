# Agent如何获取工具列表

## MCP协议标准方法

在MCP (Model Context Protocol) 协议中，Agent通过标准的 `tools/list` 方法获取工具列表。这是MCP协议的内置方法，不需要定义，由FastMCP自动实现。

## 1. Agent获取工具列表的流程

### 标准MCP方法调用

Agent连接到MCP服务器后，首先会调用 `tools/list` 方法获取可用工具列表：

**请求**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**响应**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_parameter",
        "description": "获取单个参数的详细信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "parameter_name": {
              "type": "string",
              "description": "参数名称，例如 'income_tax_rate'"
            }
          },
          "required": ["parameter_name"]
        }
      },
      {
        "name": "set_parameter",
        "description": "设置单个仿真参数，自动验证合法性",
        "inputSchema": {
          "type": "object",
          "properties": {
            "parameter_name": {
              "type": "string",
              "description": "参数名称，例如 'income_tax_rate'"
            },
            "value": {
              "oneOf": [
                {"type": "number"},
                {"type": "string"},
                {"type": "boolean"}
              ],
              "description": "新的参数值（数字、布尔值或字符串）"
            },
            "validate": {
              "type": "boolean",
              "description": "是否验证参数合法性",
              "default": true
            }
          },
          "required": ["parameter_name", "value"]
        }
      },
      // ... 更多工具
    ]
  }
}
```

## 2. FastMCP如何生成工具列表

### 自动Schema生成

FastMCP自动从注册的工具生成工具列表：

1. **扫描所有注册的工具**: FastMCP扫描所有使用 `@mcp.tool()` 装饰器注册的函数
2. **提取工具信息**:
   - `name`: 函数名称
   - `description`: 从docstring提取
   - `inputSchema`: 从函数签名和类型注解自动生成
3. **生成JSON Schema**: FastMCP根据函数参数类型自动生成JSON Schema

### 示例：工具注册到列表生成

```python
# 1. 注册工具
@mcp.tool()
async def get_parameter(parameter_name: str) -> str:
    """
    获取单个参数的详细信息
    
    Args:
        parameter_name: 参数名称，例如 'income_tax_rate'
    
    Returns:
        参数详细信息（JSON字符串）
    """
    result = param_manager.get_parameter(parameter_name)
    return json.dumps(result, indent=2, ensure_ascii=False)

# 2. FastMCP自动生成工具列表
# FastMCP会:
# - 提取函数名: "get_parameter"
# - 提取描述: "获取单个参数的详细信息"
# - 解析参数: parameter_name: str
# - 生成Schema: {"type": "object", "properties": {"parameter_name": {"type": "string"}}}
```

## 3. Agent初始化时的工具发现

### 连接建立后的标准流程

当Agent首次连接到MCP服务器时，标准流程是：

```
1. 建立连接（STDIO/SSE/HTTP）
   ↓
2. 调用 tools/list 获取工具列表
   ↓
3. 缓存工具列表和Schema
   ↓
4. 根据用户输入选择合适的工具
   ↓
5. 调用 tools/call 执行工具
```

### 实际调用示例

```python
# Agent伪代码
class MCPAgent:
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        self.tools = []  # 工具缓存
    
    async def connect(self):
        # 1. 建立连接
        await self.mcp_server.connect()
        
        # 2. 获取工具列表
        response = await self.mcp_server.call("tools/list", {})
        self.tools = response["result"]["tools"]
        
        print(f"已发现 {len(self.tools)} 个工具:")
        for tool in self.tools:
            print(f"  - {tool['name']}: {tool['description']}")
    
    def find_tool(self, task_description):
        # 3. 根据任务描述选择合适的工具
        # LLM会根据工具描述和任务匹配
        for tool in self.tools:
            if self._matches_task(tool, task_description):
                return tool
        return None
    
    async def execute_task(self, task_description):
        # 4. 选择合适的工具
        tool = self.find_tool(task_description)
        if not tool:
            return "未找到合适的工具"
        
        # 5. 构建参数（LLM根据工具schema和任务描述生成）
        arguments = self._build_arguments(tool, task_description)
        
        # 6. 调用工具
        result = await self.mcp_server.call("tools/call", {
            "name": tool["name"],
            "arguments": arguments
        })
        
        return result
```

## 4. MCP协议的标准方法

MCP协议定义了以下标准方法，不需要在服务器端实现，由MCP框架自动处理：

### 标准方法列表

1. **tools/list**: 列出所有可用工具
   - Agent调用此方法获取工具列表
   - 服务器自动返回所有注册的工具

2. **tools/call**: 调用指定工具
   - Agent调用此方法执行工具
   - 服务器路由到对应的函数并执行

3. **prompts/list**: 列出所有可用提示
   - 获取可用的提示模板（如果支持）

4. **prompts/get**: 获取特定提示
   - 获取提示模板内容（如果支持）

5. **resources/list**: 列出所有可用资源
   - 获取可用的资源列表（如果支持）

6. **resources/read**: 读取特定资源
   - 读取资源内容（如果支持）

## 5. 查看工具列表的方法

### 方法1: Agent自动调用（标准方式）

Agent连接后会自动调用 `tools/list`：

```python
# Agent自动执行
response = await mcp_client.call("tools/list", {})
tools = response["result"]["tools"]
```

### 方法2: 服务器启动时打印（调试方式）

服务器启动时会打印所有工具（用于调试）：

```bash
python server_fastmcp.py

# 输出:
# ✅ AI经济学家参数服务器已初始化
# 📋 参数管理工具 (10个):
#   1. get_all_parameters    - 获取所有参数
#   2. get_parameter         - 获取单个参数
#   ...
```

### 方法3: 使用MCP客户端测试（手动方式）

可以使用MCP客户端手动调用：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_tools():
    async with stdio_client(StdioServerParameters(
        command="python",
        args=["server_fastmcp.py"]
    )) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 获取工具列表
            result = await session.list_tools()
            print(f"发现 {len(result.tools)} 个工具:")
            for tool in result.tools:
                print(f"  - {tool.name}: {tool.description}")
```

## 6. 工具列表的结构

### 工具对象结构

每个工具对象包含以下字段：

```typescript
interface Tool {
  name: string;                    // 工具名称（函数名）
  description: string;             // 工具描述（docstring）
  inputSchema: {                   // JSON Schema
    type: "object";
    properties: {
      [param_name: string]: {
        type: string;              // 参数类型（string/number/boolean）
        description?: string;      // 参数描述
        default?: any;             // 默认值（如果有）
      }
    };
    required?: string[];           // 必需参数列表
  };
}
```

### 示例：完整的工具对象

```json
{
  "name": "analyze_question",
  "description": "分析问题并识别实验类型\n\n识别的问题类型包括：\n- innovation: 创新促进政策\n- redistribution: 全民基本收入/再分配政策\n- labor_productivity: AI/自动化对劳动力市场的影响\n- tariff: 关税/税收政策冲击\n\nArgs:\n    question: 问题文本\n\nReturns:\n    问题分析结果（JSON字符串）",
  "inputSchema": {
    "type": "object",
    "properties": {
      "question": {
        "type": "string",
        "description": "问题文本，例如：\n- \"How do innovation-promoting policies shape economic performance?\"\n- \"How will a universal basic income policy affect people's lives?\""
      }
    },
    "required": ["question"]
  }
}
```

## 7. 工具列表的缓存

### Agent缓存机制

大多数MCP客户端（如Cursor）会缓存工具列表：

1. **首次连接**: Agent调用 `tools/list` 获取工具列表
2. **缓存列表**: 将工具列表缓存在内存中
3. **后续使用**: 直接从缓存查找工具，不需要重复调用
4. **更新机制**: 如果服务器重启或工具变更，Agent会重新获取列表

### 何时重新获取工具列表

Agent会在以下情况重新获取工具列表：

1. **服务器重启**: 连接断开后重新连接
2. **工具变更**: 服务器通知工具列表已更新（如果支持）
3. **手动刷新**: 用户手动刷新工具列表
4. **定期更新**: 某些客户端会定期检查工具列表更新

## 总结

1. **标准方法**: Agent通过 `tools/list` 方法获取工具列表（MCP协议标准方法）
2. **自动生成**: FastMCP自动从注册的工具生成工具列表和Schema
3. **初始化流程**: Agent连接后首先调用 `tools/list` 获取工具列表
4. **缓存机制**: Agent会缓存工具列表以提高效率
5. **无需定义**: `tools/list` 是MCP协议标准方法，不需要在服务器端实现

整个工具发现机制是完全自动化的：
- 服务器端：只需使用 `@mcp.tool()` 注册工具
- 客户端：只需调用 `tools/list` 获取工具列表
- FastMCP：自动处理所有细节

