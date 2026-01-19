#!/usr/bin/env python3
"""
AI经济学家 MCP Server (FastMCP版本)

使用FastMCP简化版本，代码更简洁，自动生成Schema
相比传统Server版本，代码量减少约60%

功能模块：
1. 参数管理工具 - 配置仿真参数
2. 仿真控制工具 - 启动、停止、查询仿真
3. 历史实验分析工具 - 读取和分析历史实验记录

用法:
    # STDIO模式（默认，用于本地Cursor连接）
    python server_fastmcp.py
    
    # SSE模式（用于远程SSH访问）
    python server_fastmcp.py --transport sse --port 8000
    
    # Streamable HTTP模式（用于远程SSH访问）
    python server_fastmcp.py --transport streamable-http --port 8000
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP

# 设置 MCP_MODE 环境变量，告诉所有模块使用 CPU
os.environ['MCP_MODE'] = '1'

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入完整的 SimulationConfig
from agentsociety_ecosim.simulation.joint_debug_test import SimulationConfig

# 导入参数管理器
from parameter_manager import ParameterManager

# 导入工具模块
from tools import history_tools

# 尝试导入仿真工具（可选，如果失败不影响历史分析功能）
try:
    from tools import simulation_tools
    SIMULATION_TOOLS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  仿真工具不可用: {e}")
    simulation_tools = None
    SIMULATION_TOOLS_AVAILABLE = False

# 导入实验分析工具
try:
    from tools import experiment_analyzer_tools
    ANALYZER_TOOLS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  实验分析工具不可用: {e}")
    experiment_analyzer_tools = None
    ANALYZER_TOOLS_AVAILABLE = False

# 导入自动化实验工具
try:
    from tools import automated_experiment_tools
    AUTOMATED_TOOLS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  自动化实验工具不可用: {e}")
    automated_experiment_tools = None
    AUTOMATED_TOOLS_AVAILABLE = False


# ========== 初始化 FastMCP ==========

mcp = FastMCP("ai-economist-parameter-server")

# 创建完整的配置对象
config = SimulationConfig()

# 初始化参数管理器
ParameterManager.reset_instance()
param_manager = ParameterManager.get_instance(config=config)

print("✅ AI经济学家参数服务器已初始化")
print(f"   - 加载了 {len(param_manager.metadata)} 个参数")
print(f"   - 配置对象: {type(config).__name__}")


# ========== 定义工具 (使用装饰器) ==========

@mcp.tool()
async def get_all_parameters(
    category: str = "all",
    format: str = "json"
) -> str:
    """
    获取经济仿真系统的所有可配置参数，按类别分组

    Args:
        category: 参数类别过滤 (all/tax_policy/production/labor_market/market/system_scale/redistribution/performance/monitoring)
        format: 输出格式 (json/markdown/table)

    Returns:
        参数配置数据（JSON字符串）
    """
    result = param_manager.get_all_parameters(category=category, format=format)

    if isinstance(result, str):
        return result
    else:
        return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_parameter(parameter_name: str) -> str:
    """
    获取单个参数的详细信息

    Args:
        parameter_name: 参数名称，例如 'income_tax_rate'

    Returns:
        参数详细信息（JSON字符串）
    """
    try:
        result = param_manager.get_parameter(parameter_name)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def set_parameter(
    parameter_name: str,
    value: float | int | str | bool,
    validate: bool = True
) -> str:
    """
    设置单个仿真参数，自动验证合法性

    Args:
        parameter_name: 参数名称，例如 'income_tax_rate'
        value: 新的参数值（数字、布尔值或字符串）
        validate: 是否验证参数合法性

    Returns:
        设置结果（JSON字符串）
    """
    result = param_manager.set_parameter(parameter_name, value, validate=validate)

    response = {
        "success": result.success,
        "valid": result.valid,
        "old_value": result.old_value,
        "new_value": result.new_value,
        "warnings": result.warnings,
        "errors": result.errors
    }

    return json.dumps(response, indent=2, ensure_ascii=False)


@mcp.tool()
async def batch_set_parameters(
    parameters: Dict[str, Any],
    scenario_name: Optional[str] = None
) -> str:
    """
    批量设置多个参数（用于场景设置）

    Args:
        parameters: 参数键值对字典
        scenario_name: 场景名称（可选，用于保存预设）

    Returns:
        批量设置结果（JSON字符串）
    """
    results = param_manager.batch_set_parameters(parameters, scenario_name=scenario_name)

    # 转换ValidationResult为可序列化的字典
    serializable_results = {}
    for param_name, result in results.items():
        serializable_results[param_name] = {
            "success": result.success,
            "valid": result.valid,
            "old_value": result.old_value,
            "new_value": result.new_value,
            "warnings": result.warnings,
            "errors": result.errors
        }

    response = {
        "success": all(r.success for r in results.values()),
        "updated_count": sum(1 for r in results.values() if r.success),
        "failed_count": sum(1 for r in results.values() if not r.success),
        "scenario_name": scenario_name,
        "details": serializable_results
    }

    return json.dumps(response, indent=2, ensure_ascii=False)


@mcp.tool()
async def validate_parameters(parameters: Dict[str, Any]) -> str:
    """
    验证参数配置是否合法（不实际修改配置）

    Args:
        parameters: 要验证的参数键值对字典

    Returns:
        验证结果（JSON字符串）
    """
    result = param_manager.validate_parameters(parameters)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
async def reset_parameters(parameters: Optional[List[str]] = None) -> str:
    """
    重置参数为默认值

    Args:
        parameters: 要重置的参数列表（为空表示重置所有）

    Returns:
        重置结果消息
    """
    param_manager.reset_parameters(parameters)
    return f"已重置 {len(parameters) if parameters else '所有'} 个参数为默认值"


@mcp.tool()
async def save_preset(
    name: str,
    description: str = "",
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    保存当前参数配置为预设

    Args:
        name: 预设名称
        description: 预设描述
        parameters: 要保存的参数（为空表示保存当前所有参数）

    Returns:
        保存结果消息
    """
    if not parameters:
        # 保存当前所有参数
        parameters = {}
        for param_name in param_manager.metadata.keys():
            param_info = param_manager.get_parameter(param_name)
            parameters[param_name] = param_info["value"]

    param_manager.save_preset(name, parameters, description)
    return f"预设 '{name}' 已保存，包含 {len(parameters)} 个参数"


@mcp.tool()
async def load_preset(name: str, apply: bool = False) -> str:
    """
    加载参数预设

    Args:
        name: 预设名称
        apply: 是否立即应用到当前配置

    Returns:
        预设内容或应用结果（JSON字符串）
    """
    try:
        preset = param_manager.load_preset(name)

        if apply:
            # 应用到当前配置
            results = param_manager.batch_set_parameters(preset.parameters)
            success_count = sum(1 for r in results.values() if r.success)

            response = {
                "success": True,
                "message": f"预设 '{name}' 已加载并应用",
                "success_count": success_count,
                "total_count": len(preset.parameters)
            }
            return json.dumps(response, indent=2, ensure_ascii=False)
        else:
            # 只返回预设内容
            response = {
                "name": preset.name,
                "description": preset.description,
                "created_at": preset.created_at,
                "parameters": preset.parameters
            }
            return json.dumps(response, indent=2, ensure_ascii=False)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def list_presets() -> str:
    """
    列出所有可用的参数预设

    Returns:
        预设列表（JSON字符串）
    """
    presets = param_manager.list_presets()
    return json.dumps(presets, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_parameter_ranges() -> str:
    """
    获取所有参数的合法取值范围

    Returns:
        参数范围信息（JSON字符串）
    """
    ranges = param_manager.get_parameter_ranges()
    return json.dumps(ranges, indent=2, ensure_ascii=False)

@mcp.tool()
def list_yaml_configs() -> str:
    """
    列出所有可用的YAML配置文件
    
    Returns:
        可用配置列表（JSON格式）
    """
    configs = param_manager.list_yaml_configs()
    return json.dumps({
        "success": True,
        "count": len(configs),
        "configs": configs
    }, indent=2, ensure_ascii=False)

@mcp.tool()
def load_yaml_config(config_name: str) -> str:
    """
    从YAML文件加载配置并应用
    
    Args:
        config_name: 配置文件名（不含.yaml后缀），例如 "default", "high_tax_scenario"
        
    Returns:
        加载和应用结果（JSON格式）
        
    Example:
        load_yaml_config("high_tax_scenario")
    """
    result = param_manager.apply_yaml_config(config_name, validate=True)
    return json.dumps(result, indent=2, ensure_ascii=False)

@mcp.tool()
def save_current_config_to_yaml(config_name: str, description: str = "") -> str:
    """
    将当前配置保存为YAML文件
    
    Args:
        config_name: 配置文件名（不含.yaml后缀）
        description: 配置描述
        
    Returns:
        保存结果（JSON格式）
        
    Example:
        save_current_config_to_yaml("my_custom_config", "Custom configuration for testing")
    """
    result = param_manager.save_current_config_to_yaml(config_name, description)
    return json.dumps(result, indent=2, ensure_ascii=False)


# ========== 注册干预控制工具 ==========

print("🎛️  注册干预控制工具...")

from agentsociety_ecosim.mcp_server.tools.intervention_tools import (
    pause_simulation_tool,
    resume_simulation_tool,
    inject_intervention_tool,
    list_pending_interventions_tool,
    cancel_intervention_tool
)

@mcp.tool()
async def pause_simulation() -> str:
    """
    暂停正在运行的仿真
    
    功能：立即暂停仿真执行，保持当前状态
    
    Returns:
        暂停结果（JSON格式）
    """
    return await pause_simulation_tool()


@mcp.tool()
async def resume_simulation() -> str:
    """
    恢复已暂停的仿真
    
    功能：从暂停点继续执行仿真
    
    Returns:
        恢复结果（JSON格式）
    """
    return await resume_simulation_tool()


@mcp.tool()
async def inject_intervention(
    intervention_type: str,
    target_month: int,
    parameters: str,  # JSON字符串
    description: str = ""
) -> str:
    """
    向仿真注入干预措施
    
    Args:
        intervention_type: 干预类型 (parameter_change, policy, shock, injection)
        target_month: 目标月份（必须大于当前月份）
        parameters: 干预参数（JSON字符串），例如: '{"income_tax_rate": 0.30}'
        description: 干预描述
        
    Returns:
        干预调度结果（JSON格式）
        
    Example:
        inject_intervention(
            "parameter_change",
            5,
            '{"income_tax_rate": 0.35, "vat_rate": 0.25}',
            "增税政策实验"
        )
    """
    try:
        params_dict = json.loads(parameters)
    except json.JSONDecodeError:
        return json.dumps({
            "success": False,
            "message": "Invalid JSON in parameters"
        })
    
    return await inject_intervention_tool(
        intervention_type=intervention_type,
        target_month=target_month,
        parameters=params_dict,
        description=description
    )


@mcp.tool()
async def list_pending_interventions() -> str:
    """
    列出所有待执行的干预
    
    Returns:
        待执行干预列表（JSON格式）
    """
    return await list_pending_interventions_tool()


@mcp.tool()
async def cancel_intervention(intervention_id: str) -> str:
    """
    取消指定的干预
    
    Args:
        intervention_id: 干预ID
        
    Returns:
        取消结果（JSON格式）
    """
    return await cancel_intervention_tool(intervention_id)

print("✅ 干预控制工具注册完成（5个工具）")


# ========== 注册仿真工具 ==========

if SIMULATION_TOOLS_AVAILABLE:
    print("📊 注册仿真控制工具...")
    try:
        simulation_tools.register_tools(mcp, parameter_manager=param_manager)
        print("✅ 仿真工具注册完成")
    except Exception as e:
        print(f"❌ 仿真工具注册失败: {e}")
        import traceback
        traceback.print_exc()
        SIMULATION_TOOLS_AVAILABLE = False
else:
    print("⏭️  跳过仿真工具注册（仿真工具不可用）")

# ========== 注册历史实验分析工具 ==========

print("📚 注册历史实验分析工具...")
history_tools.register_tools(mcp)
print("✅ 历史实验分析工具注册完成")


# ========== 注册自动化实验工具 ==========

if AUTOMATED_TOOLS_AVAILABLE:
    print("🤖 注册自动化实验工具...")
    try:
        automated_tools = automated_experiment_tools.get_automated_tools()
        
        @mcp.tool()
        async def analyze_question(question: str) -> str:
            """
            分析问题并识别实验类型
            
            识别的问题类型包括：
            - innovation: 创新促进政策
            - redistribution: 全民基本收入/再分配政策
            - labor_productivity: AI/自动化对劳动力市场的影响
            - tariff: 关税/税收政策冲击
            
            Args:
                question: 问题文本，例如：
                    - "How do innovation-promoting policies shape economic performance?"
                    - "How will a universal basic income policy affect people's lives?"
                    - "How will AI agents reshape the labor market?"
                    - "How will a breaking news event such as the Liberation Day tariff affect the stock market?"
            
            Returns:
                问题分析结果（JSON字符串），包含问题类型、关键词、推荐的参数配置
            """
            from dataclasses import asdict
            result = automated_tools.analyze_question(question)
            return json.dumps(asdict(result), indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def generate_config_from_question(
            question: str,
            base_config_name: Optional[str] = None
        ) -> str:
            """
            根据问题自动生成实验配置
            
            此工具会自动：
            1. 分析问题类型
            2. 推荐相关参数配置
            3. 生成配置文件名
            4. 提供配置指导
            
            Args:
                question: 问题文本
                base_config_name: 基础配置名称（可选，如果提供则基于此配置修改）
            
            Returns:
                配置信息（JSON字符串），包含推荐的参数和配置名称
            """
            result = automated_tools.generate_config_from_question(question, base_config_name)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def get_experiment_workflow(question: str) -> str:
            """
            获取完整的实验工作流指导
            
            提供从问题分析到结果获取的完整步骤指导，包括：
            1. 问题分析
            2. 生成配置文件
            3. 加载配置
            4. 启动仿真
            5. 监控状态
            6. 捕捉实验
            7. 分析实验
            8. 获取结果
            
            Args:
                question: 问题文本
            
            Returns:
                完整工作流（JSON字符串），包含每个步骤的工具调用和方法
            """
            workflow = automated_tools.get_experiment_workflow(question)
            return json.dumps(workflow, indent=2, ensure_ascii=False)
        
        print("✅ 自动化实验工具注册完成（3个工具）")
    except Exception as e:
        print(f"❌ 自动化实验工具注册失败: {e}")
        import traceback
        traceback.print_exc()
        AUTOMATED_TOOLS_AVAILABLE = False
else:
    print("⏭️  跳过自动化实验工具注册（模块不可用）")


# ========== 注册实验分析工具（新） ==========

if ANALYZER_TOOLS_AVAILABLE:
    print("📊 注册实验分析工具...")
    try:
        analyzer_tools = experiment_analyzer_tools.get_analyzer_tools()
        
        @mcp.tool()
        async def capture_experiment(
            experiment_name: str,
            experiment_dir: Optional[str] = None,
            status: str = "pending"
        ) -> str:
            """
            捕捉实验目录并保存到manifest
            
            当仿真程序运行时会创建一个类似 exp_100h_12m_20251121_221420 的实验目录。
            使用此工具捕捉实验目录，保存到manifest中，便于后续分析。
            
            Args:
                experiment_name: 实验名称（如 exp_100h_12m_20251121_221420）
                experiment_dir: 实验目录路径（如果为None，则使用默认output目录）
                status: 实验状态（pending, running, completed）
            
            Returns:
                操作结果（JSON字符串）
            """
            result = analyzer_tools.capture_experiment(experiment_name, experiment_dir, status)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def update_experiment_status(experiment_name: str, status: str) -> str:
            """
            更新实验状态
            
            Args:
                experiment_name: 实验名称
                status: 新状态（pending, running, completed）
            
            Returns:
                操作结果（JSON字符串）
            """
            result = analyzer_tools.update_experiment_status(experiment_name, status)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def analyze_experiment(
            experiment_name: str,
            innovation_types: Optional[List[str]] = None,
            include_innovation: bool = True
        ) -> str:
            """
            分析实验数据
            
            分析指标包括：
            - 微观指标：创新后n个月市场占有比例增量相关系数（n=1,2,3），每个企业商品的产量、质量
            - 宏观指标：消费者购买商品属性值和，GDP
            
            Args:
                experiment_name: 实验名称
                innovation_types: 创新类型列表（如果为None，默认包含主要类型：labor_productivity_factor, price, profit_margin）
                include_innovation: 是否包含创新分析
            
            Returns:
                分析结果（JSON字符串），包含：
                - macro_metrics: GDP, total_revenue, total_expenditure, consumer total_attribute_value
                - micro_metrics: 企业商品产量和质量
                - innovation_metrics: 创新与市场占有率的相关性（如果include_innovation=True）
            """
            result = analyzer_tools.analyze_experiment(
                experiment_name, 
                innovation_types=innovation_types,
                include_innovation=include_innovation
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def list_experiments() -> str:
            """
            列出所有已捕捉的实验
            
            Returns:
                实验列表（JSON字符串），包含每个实验的名称、目录、状态等信息
            """
            result = analyzer_tools.list_experiments()
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def get_analysis_result(experiment_name: str) -> str:
            """
            获取实验分析结果
            
            Args:
                experiment_name: 实验名称
            
            Returns:
                分析结果（JSON字符串），包含完整的分析数据
            """
            result = analyzer_tools.get_analysis_result(experiment_name)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        print("✅ 实验分析工具注册完成（5个工具）")
    except Exception as e:
        print(f"❌ 实验分析工具注册失败: {e}")
        import traceback
        traceback.print_exc()
        ANALYZER_TOOLS_AVAILABLE = False
else:
    print("⏭️  跳过实验分析工具注册（模块不可用）")


# ========== 启动服务器 ==========

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="AI经济学家 MCP 服务器")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="传输模式: stdio (本地), sse (HTTP/SSE), streamable-http"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP服务器端口（仅用于sse和streamable-http模式，默认8000）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP服务器主机地址（仅用于sse和streamable-http模式，默认0.0.0.0）"
    )
    parser.add_argument(
        "--mount-path",
        type=str,
        default="/mcp",
        help="SSE挂载路径（仅用于sse模式，默认/mcp）"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 启动AI经济学家 MCP 服务器 (FastMCP版)...")
    print("=" * 60)
    print("📋 参数管理工具 (10个):")
    print("  1. get_all_parameters    - 获取所有参数")
    print("  2. get_parameter         - 获取单个参数")
    print("  3. set_parameter         - 设置单个参数")
    print("  4. batch_set_parameters  - 批量设置参数")
    print("  5. validate_parameters   - 验证参数配置")
    print("  6. reset_parameters      - 重置参数")
    print("  7. save_preset           - 保存参数预设")
    print("  8. load_preset           - 加载参数预设")
    print("  9. list_presets          - 列出所有预设")
    print(" 10. get_parameter_ranges  - 获取参数范围")
    print()

    if SIMULATION_TOOLS_AVAILABLE:
        print("🎮 仿真控制工具 (7个):")
        print("  1. start_simulation           - 启动仿真")
        print("  2. get_simulation_status      - 查询仿真状态")
        print("  3. stop_simulation            - 停止仿真")
        print("  4. get_economic_indicators    - 获取经济指标（单月）")
        print("  5. get_all_economic_indicators- 获取经济指标（多月）")
        print("  6. get_household_summary      - 获取家庭摘要")
        print("  7. get_firm_summary           - 获取企业摘要")
        print()

    print("📚 历史实验分析工具 (6个):")
    print("  1. list_history_experiments   - 列出历史实验")
    print("  2. generate_experiment_report - 生成实验报告")
    print("  3. get_experiment_summary     - 获取实验摘要")
    print("  4. get_experiment_timeseries  - 获取时间序列")
    print("  5. compare_experiments        - 对比实验")
    print("  6. get_monthly_statistics     - 读取月度统计")
    print()

    if ANALYZER_TOOLS_AVAILABLE:
        print("📊 实验分析工具 (5个):")
        print("  1. capture_experiment      - 捕捉实验目录")
        print("  2. update_experiment_status- 更新实验状态")
        print("  3. analyze_experiment      - 分析实验数据")
        print("  4. list_experiments        - 列出已捕捉实验")
        print("  5. get_analysis_result     - 获取分析结果")
        print()

    if AUTOMATED_TOOLS_AVAILABLE:
        print("🤖 自动化实验工具 (3个):")
        print("  1. analyze_question            - 分析问题类型")
        print("  2. generate_config_from_question - 根据问题生成配置")
        print("  3. get_experiment_workflow     - 获取完整工作流指导")
        print()

    tool_count = 10 + (7 if SIMULATION_TOOLS_AVAILABLE else 0) + 6 + (5 if ANALYZER_TOOLS_AVAILABLE else 0) + (3 if AUTOMATED_TOOLS_AVAILABLE else 0)
    print("=" * 60)
    # ==================== YAML配置管理工具 ====================
    
   
    # 注册实验分析工具
    if ANALYZER_TOOLS_AVAILABLE:
        analyzer_tools = experiment_analyzer_tools.get_analyzer_tools()
        
        @mcp.tool()
        async def capture_experiment(
            experiment_name: str,
            experiment_dir: Optional[str] = None,
            status: str = "pending"
        ) -> str:
            """
            捕捉实验目录并保存到manifest
            
            Args:
                experiment_name: 实验名称（如 exp_100h_12m_20251121_221420）
                experiment_dir: 实验目录路径（如果为None，则使用默认output目录）
                status: 实验状态（pending, running, completed）
            
            Returns:
                操作结果（JSON字符串）
            """
            result = analyzer_tools.capture_experiment(experiment_name, experiment_dir, status)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def update_experiment_status(experiment_name: str, status: str) -> str:
            """
            更新实验状态
            
            Args:
                experiment_name: 实验名称
                status: 新状态（pending, running, completed）
            
            Returns:
                操作结果（JSON字符串）
            """
            result = analyzer_tools.update_experiment_status(experiment_name, status)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def analyze_experiment(
            experiment_name: str,
            innovation_types: Optional[List[str]] = None,
            include_innovation: bool = True
        ) -> str:
            """
            分析实验数据
            
            分析指标包括：
            - 微观指标：创新后n个月市场占有比例增量相关系数（n=1,2,3），每个企业商品的产量、质量
            - 宏观指标：消费者购买商品属性值和，GDP
            
            Args:
                experiment_name: 实验名称
                innovation_types: 创新类型列表（如果为None，默认包含主要类型：labor_productivity_factor, price, profit_margin）
                include_innovation: 是否包含创新分析
            
            Returns:
                分析结果（JSON字符串），包含：
                - macro_metrics: GDP, total_revenue, total_expenditure, consumer total_attribute_value
                - micro_metrics: 企业商品产量和质量
                - innovation_metrics: 创新与市场占有率的相关性（如果include_innovation=True）
            """
            result = analyzer_tools.analyze_experiment(
                experiment_name, 
                innovation_types=innovation_types,
                include_innovation=include_innovation
            )
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def list_experiments() -> str:
            """
            列出所有已捕捉的实验
            
            Returns:
                实验列表（JSON字符串）
            """
            result = analyzer_tools.list_experiments()
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        @mcp.tool()
        async def get_analysis_result(experiment_name: str) -> str:
            """
            获取实验分析结果
            
            Args:
                experiment_name: 实验名称
            
            Returns:
                分析结果（JSON字符串）
            """
            result = analyzer_tools.get_analysis_result(experiment_name)
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        print("✅ 实验分析工具注册完成（5个工具）")
   
    print(f"\n✨ 总计 {tool_count + 3} 个工具已就绪（含3个YAML配置工具）")
    
    # 根据传输模式启动服务器
    if args.transport == "stdio":
        print("📡 传输模式: STDIO (标准输入输出)")
        print("🔌 等待客户端连接...\n")
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        print(f"📡 传输模式: SSE (Server-Sent Events)")
        print(f"🌐 服务器地址: http://{args.host}:{args.port}{args.mount_path}")
        print(f"🔌 等待HTTP连接...")
        print(f"\n💡 Cursor配置:")
        print(f'   "url": "http://localhost:{args.port}{args.mount_path}"')
        print(f'   "transport": "sse"')
        print(f"\n💡 SSH端口转发命令:")
        print(f"   ssh -L {args.port}:localhost:{args.port} user@remote-server\n")
        mcp.run(transport="sse", mount_path=args.mount_path)
    elif args.transport == "streamable-http":
        print(f"📡 传输模式: Streamable HTTP")
        print(f"🌐 服务器地址: http://{args.host}:{args.port}/mcp")
        print(f"🔌 等待HTTP连接...")
        print(f"\n💡 Cursor配置:")
        print(f'   "url": "http://localhost:{args.port}/mcp"')
        print(f'   "transport": "streamable"')
        print(f"\n💡 SSH端口转发命令:")
        print(f"   ssh -L {args.port}:localhost:{args.port} user@remote-server\n")
        mcp.run(transport="streamable-http")
