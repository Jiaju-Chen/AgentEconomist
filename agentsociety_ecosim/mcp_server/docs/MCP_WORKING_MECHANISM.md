# MCP工具调用机制详解

## 概述

MCP (Model Context Protocol) 是一个标准化的协议，用于AI Agent与外部工具/服务进行交互。本文档说明当前实现中MCP如何工作、如何路由工具调用。

## 架构概览

```
Agent (Cursor/LLM)
    ↓
MCP Protocol (JSON-RPC over STDIO/SSE/HTTP)
    ↓
FastMCP Server (server_fastmcp.py)
    ↓
Tool Functions (使用 @mcp.tool() 装饰器注册)
    ↓
业务逻辑 (parameter_manager, experiment_analyzer等)
```

## 1. 工具注册机制

### FastMCP装饰器模式

所有工具都使用 `@mcp.tool()` 装饰器注册：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-economist-parameter-server")

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
```

### 自动Schema生成

FastMCP会自动：
1. **提取函数签名**: 从函数参数和类型注解提取工具定义
2. **提取文档字符串**: 从docstring提取工具描述和参数说明
3. **生成JSON Schema**: 自动生成符合MCP协议的工具schema

### 工具注册示例

```python
# 工具通过函数名识别
@mcp.tool()
async def get_parameter(parameter_name: str) -> str:
    """工具描述"""
    # 实现
    pass

# 函数名 "get_parameter" 就是工具名称
# Agent调用时使用 "get_parameter" 作为tool_name
```

## 2. Agent如何调用工具

### MCP协议消息格式

Agent通过MCP协议发送工具调用请求，格式如下：

**请求消息**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_parameter",
    "arguments": {
      "parameter_name": "income_tax_rate"
    }
  }
}
```

**响应消息**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"name\": \"income_tax_rate\", \"value\": 0.45, ...}"
      }
    ]
  }
}
```

### 工具列表获取

Agent首先会调用 `tools/list` 获取可用工具列表：

**请求**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
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
      // ... 更多工具
    ]
  }
}
```

## 3. MCP如何路由工具调用

### FastMCP的内部机制

1. **工具注册阶段**:
   - `@mcp.tool()` 装饰器将函数注册到内部的工具字典
   - 工具名称 = 函数名称
   - 保存函数引用和schema信息

2. **请求接收阶段**:
   - MCP服务器接收到 `tools/call` 请求
   - 提取 `params.name` (工具名称)
   - 在工具字典中查找对应的函数

3. **参数解析阶段**:
   - 从 `params.arguments` 提取参数
   - 根据函数签名和类型注解验证参数
   - 转换参数类型（如果需要）

4. **函数执行阶段**:
   - 调用注册的函数
   - 传入解析后的参数
   - 捕获返回值

5. **响应返回阶段**:
   - 将函数返回值包装成MCP响应格式
   - 返回给Agent

### 路由流程图

```
Agent发送请求
    ↓
FastMCP接收 tools/call 请求
    ↓
提取 tool_name (如 "get_parameter")
    ↓
在注册的工具字典中查找 tool_name
    ↓
找到对应的函数 (get_parameter)
    ↓
解析参数 (parameter_name: "income_tax_rate")
    ↓
调用函数 param_manager.get_parameter("income_tax_rate")
    ↓
返回结果 JSON字符串
    ↓
包装成MCP响应格式
    ↓
返回给Agent
```

## 4. 当前系统的工具组织

### 工具分类

当前系统有多个工具类别，通过模块组织：

1. **参数管理工具** (10个):
   - `get_all_parameters`
   - `get_parameter`
   - `set_parameter`
   - `batch_set_parameters`
   - `validate_parameters`
   - `reset_parameters`
   - `save_preset`
   - `load_preset`
   - `list_presets`
   - `get_parameter_ranges`

2. **YAML配置工具** (3个):
   - `list_yaml_configs`
   - `load_yaml_config`
   - `save_current_config_to_yaml`

3. **干预控制工具** (5个):
   - `pause_simulation`
   - `resume_simulation`
   - `inject_intervention`
   - `list_pending_interventions`
   - `cancel_intervention`

4. **仿真控制工具** (7个，如果可用):
   - `start_simulation`
   - `get_simulation_status`
   - `stop_simulation`
   - `get_economic_indicators`
   - `get_all_economic_indicators`
   - `get_household_summary`
   - `get_firm_summary`

5. **历史实验分析工具** (6个):
   - `list_history_experiments`
   - `generate_experiment_report`
   - `get_experiment_summary`
   - `get_experiment_timeseries`
   - `compare_experiments`
   - `get_monthly_statistics`

6. **实验分析工具** (5个):
   - `capture_experiment`
   - `update_experiment_status`
   - `analyze_experiment`
   - `list_experiments`
   - `get_analysis_result`

7. **自动化实验工具** (3个):
   - `analyze_question`
   - `generate_config_from_question`
   - `get_experiment_workflow`

### 工具命名规范

- 工具名称 = Python函数名称
- 使用snake_case命名
- 名称应该清晰描述功能

## 5. Agent如何使用工具

### 典型交互流程

```python
# 1. Agent获取工具列表
tools_list = call_mcp("tools/list")

# 2. Agent选择需要的工具（基于问题分析）
# 例如：需要设置参数
selected_tool = "batch_set_parameters"

# 3. Agent调用工具
result = call_mcp("tools/call", {
    "name": "batch_set_parameters",
    "arguments": {
        "parameters": {
            "innovation_probability": 0.15,
            "enable_innovation": True
        }
    }
})

# 4. Agent处理结果并继续
if result["success"]:
    # 继续下一步
    next_result = call_mcp("tools/call", {
        "name": "start_simulation",
        "arguments": {}
    })
```

### LLM驱动的工具选择

在现代AI Agent中（如Cursor），LLM会：
1. **理解用户意图**: 分析用户的问题或指令
2. **匹配工具**: 从工具列表中找到合适的工具
3. **构建参数**: 根据工具schema和用户输入构建参数
4. **调用工具**: 发送MCP请求
5. **处理结果**: 基于结果继续或调整策略

## 6. 实际示例

### 示例1: Agent自动配置实验

```
用户: "How will AI agents reshape the labor market?"

Agent思考流程:
1. 分析问题 → 调用 analyze_question("How will AI agents...")
   - 返回: question_type="labor_productivity", recommended_parameters={...}
   
2. 生成配置 → 调用 generate_config_from_question(...)
   - 返回: config_name="labor_productivity_policy_20251122_101542"
   
3. 设置参数 → 调用 batch_set_parameters(recommended_parameters)
   - 返回: success=True
   
4. 保存配置 → 调用 save_current_config_to_yaml(config_name, description)
   - 返回: success=True, path="..."
   
5. 加载配置 → 调用 load_yaml_config(config_name)
   - 返回: success=True
   
6. 启动仿真 → 调用 start_simulation()
   - 返回: simulation_started=True
   
7. 监控状态 → 循环调用 get_simulation_status()
   - 等待: running=False
   
8. 捕捉实验 → 调用 capture_experiment(experiment_name, "completed")
   - 返回: success=True
   
9. 分析实验 → 调用 analyze_experiment(experiment_name, include_innovation=False)
   - 返回: {macro_metrics: {...}, micro_metrics: {...}}
   
10. 得出结论 → 基于分析结果生成结论
```

### 示例2: 工具调用消息流

```
Agent → MCP Server:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "analyze_question",
    "arguments": {
      "question": "How will AI agents reshape the labor market?"
    }
  }
}

MCP Server → Agent:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"question_type\": \"labor_productivity\", \"identified_keywords\": [\"AI\", \"labor market\", \"reshap\"], ...}"
      }
    ]
  }
}
```

## 7. 扩展新工具

### 添加新工具的步骤

1. **定义函数**:
```python
@mcp.tool()
async def my_new_tool(param1: str, param2: int) -> str:
    """
    工具描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
    
    Returns:
        返回值描述
    """
    # 实现逻辑
    result = do_something(param1, param2)
    return json.dumps(result, indent=2, ensure_ascii=False)
```

2. **注册工具**:
   - FastMCP会自动注册（通过装饰器）
   - 函数名就是工具名称

3. **重启服务器**:
   - 工具会自动出现在 `tools/list` 中

## 8. 调试工具调用

### 查看可用工具

```bash
# 启动服务器时会打印所有工具
python server_fastmcp.py

# 输出示例:
# 📋 参数管理工具 (10个):
#   1. get_all_parameters    - 获取所有参数
#   2. get_parameter         - 获取单个参数
#   ...
```

### 测试工具调用

可以使用MCP客户端测试工具调用，或查看服务器日志。

## 总结

1. **工具注册**: 通过 `@mcp.tool()` 装饰器自动注册
2. **工具路由**: FastMCP根据工具名称（函数名）路由到对应函数
3. **Agent调用**: 通过MCP协议发送 `tools/call` 请求
4. **参数解析**: FastMCP自动解析和验证参数
5. **结果返回**: 函数返回值自动包装成MCP响应格式

整个机制是自动化的，开发者只需：
- 定义函数
- 添加装饰器
- 编写文档字符串

FastMCP会处理其余所有工作。

