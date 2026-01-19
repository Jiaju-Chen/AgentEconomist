"""
Tool Manager for Design Agent
统一管理所有工具的描述、分类、使用示例
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import inspect
import re


class ToolCategory(Enum):
    """工具分类"""
    CONFIG_MANAGEMENT = "配置管理"
    EXPERIMENT_EXECUTION = "实验执行"
    DATA_ANALYSIS = "数据分析"
    PROJECT_MANAGEMENT = "项目管理"


@dataclass
class ToolParameter:
    """工具参数描述"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    enabled: bool = True


class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """注册所有工具"""
        # 0. Parameter query tools (should be called BEFORE designing experiments)
        self.register_tool(
            ToolInfo(
                name="get_available_parameters",
                description="获取仿真系统中所有可用的参数，按类别分组。在设计实验之前应该调用此工具来发现所有可用参数，不要假设只知道 innovation 参数",
                category=ToolCategory.CONFIG_MANAGEMENT,
                parameters=[
                    ToolParameter("category", "str", "参数类别过滤 (all/tax_policy/production/labor_market/market/system_scale/redistribution/performance/monitoring/innovation)，默认 'all'", required=False, default="all"),
                ],
                examples=[
                    "get_available_parameters(category='all')  # 获取所有参数",
                    "get_available_parameters(category='tax_policy')  # 获取税收政策相关参数",
                    "get_available_parameters(category='innovation')  # 获取创新相关参数",
                ],
                notes=[
                    "⚠️ 重要：在设计实验前必须调用此工具来发现所有可用参数",
                    "不要假设只知道 innovation 参数，系统中有很多其他参数类别",
                    "参数类别包括：tax_policy, production, labor_market, market, system_scale, redistribution, performance, monitoring, innovation"
                ]
            )
        )
        
        self.register_tool(
            ToolInfo(
                name="get_parameter_info",
                description="获取单个参数的详细信息，包括当前值、允许范围、类型和描述。在修改参数前可以用此工具确认参数详情",
                category=ToolCategory.CONFIG_MANAGEMENT,
                parameters=[
                    ToolParameter("parameter_name", "str", "参数名称，使用点号分隔格式，例如 'innovation.policy_encourage_innovation' 或 'tax_policy.income_tax_rate'", required=True),
                ],
                examples=[
                    "get_parameter_info(parameter_name='innovation.policy_encourage_innovation')",
                    "get_parameter_info(parameter_name='tax_policy.income_tax_rate')",
                ],
                notes=[
                    "使用点号分隔的路径格式：category.parameter_name",
                    "如果参数不存在，会返回错误信息"
                ]
            )
        )
        
        # 1. init_experiment_manifest
        self.register_tool(
            ToolInfo(
                name="init_experiment_manifest",
                description="创建实验清单（manifest.yaml），用于记录实验的元数据、配置、运行状态和结果",
                category=ToolCategory.PROJECT_MANAGEMENT,
                parameters=[
                    ToolParameter("experiment_dir", "str", "实验目录路径（绝对或相对路径）", required=True),
                    ToolParameter("experiment_name", "str", "实验名称", required=True),
                    ToolParameter("research_question", "str", "研究问题", required=False, default=""),
                    ToolParameter("hypothesis", "str", "研究假设", required=False, default=""),
                    ToolParameter("expected_outcome", "str", "预期结果", required=False, default=""),
                ],
                examples=[
                    "init_experiment_manifest(\n"
                    "    experiment_dir='experiments/policy_study',\n"
                    "    experiment_name='Policy Innovation Study',\n"
                    "    research_question='Does government policy encourage innovation?',\n"
                    "    hypothesis='Policy support will increase innovation frequency'\n"
                    ")"
                ],
                notes=[
                    "每个研究项目应该先调用此工具创建 manifest",
                    "manifest 会自动记录实验的各个阶段状态",
                    "所有配置文件都会保存在 manifest 目录下"
                ]
            )
        )
        
        # 2. create_yaml_from_template
        self.register_tool(
            ToolInfo(
                name="create_yaml_from_template",
                description="从模板文件创建新的 YAML 配置文件，只能修改现有参数，不能创建新参数",
                category=ToolCategory.CONFIG_MANAGEMENT,
                parameters=[
                    ToolParameter("source_file", "str", "源模板文件路径（默认: default.yaml）", required=False, default="default.yaml"),
                    ToolParameter("output_dir", "str", "输出目录（可选，默认在 manifest 目录下）", required=False),
                    ToolParameter("parameter_changes", "dict", "要修改的参数字典，格式: {'参数路径': 新值}", required=True),
                    ToolParameter("custom_filename", "str", "自定义文件名（可选）", required=False),
                    ToolParameter("manifest_path", "str", "manifest.yaml 路径（用于自动记录配置）", required=False),
                    ToolParameter("config_label", "str", "配置标签（如 'control' 或 'treatment'）", required=False),
                ],
                examples=[
                    "create_yaml_from_template(\n"
                    "    source_file='default.yaml',\n"
                    "    parameter_changes={\n"
                    "        'innovation.policy_encourage_innovation': False\n"
                    "    },\n"
                    "    manifest_path='experiments/policy_study/manifest.yaml',\n"
                    "    config_label='control'\n"
                    ")"
                ],
                notes=[
                    "⚠️ 重要：只能修改模板中已存在的参数，不能创建新参数",
                    "参数路径使用点号分隔，如 'innovation.policy_encourage_innovation'",
                    "如果提供 manifest_path 和 config_label，配置会自动记录到 manifest 中"
                ]
            )
        )
        
        # 3. run_simulation
        self.register_tool(
            ToolInfo(
                name="run_simulation",
                description="运行经济仿真实验，执行指定的 YAML 配置文件",
                category=ToolCategory.EXPERIMENT_EXECUTION,
                parameters=[
                    ToolParameter("config_file", "str", "YAML 配置文件路径", required=True),
                    ToolParameter("experiment_name", "str", "实验名称（可选，默认从文件名推导）", required=False),
                    ToolParameter("timeout", "int", "超时时间（秒，默认 3600）", required=False, default=3600),
                    ToolParameter("manifest_path", "str", "manifest.yaml 路径（用于更新运行状态）", required=False),
                    ToolParameter("run_label", "str", "运行标签（如 'control' 或 'treatment'）", required=False),
                ],
                examples=[
                    "run_simulation(\n"
                    "    config_file='experiments/policy_study/control_policy.yaml',\n"
                    "    manifest_path='experiments/policy_study/manifest.yaml',\n"
                    "    run_label='control'\n"
                    ")"
                ],
                notes=[
                    "仿真运行时间可能较长，请耐心等待",
                    "运行状态会自动更新到 manifest 中",
                    "运行日志会保存在配置文件同目录下的 .log 文件中"
                ]
            )
        )
        
        # 4. read_simulation_report
        self.register_tool(
            ToolInfo(
                name="read_simulation_report",
                description="从仿真报告 JSON 文件中提取关键经济指标",
                category=ToolCategory.DATA_ANALYSIS,
                parameters=[
                    ToolParameter("report_file", "str", "报告文件路径（可选）", required=False),
                    ToolParameter("experiment_name", "str", "实验名称（如果未提供 report_file）", required=False),
                    ToolParameter("manifest_path", "str", "manifest.yaml 路径（用于保存指标）", required=False),
                    ToolParameter("run_label", "str", "运行标签（用于保存指标到对应 run）", required=False),
                ],
                examples=[
                    "read_simulation_report(\n"
                    "    experiment_name='control_policy',\n"
                    "    manifest_path='experiments/policy_study/manifest.yaml',\n"
                    "    run_label='control'\n"
                    ")"
                ],
                notes=[
                    "提取的关键指标包括：就业率、失业率、平均收入、支出、储蓄率、财富分布、基尼系数等",
                    "如果提供 manifest_path 和 run_label，指标会自动保存到 manifest 中"
                ]
            )
        )
        
        # 5. analyze_experiment_directory
        self.register_tool(
            ToolInfo(
                name="analyze_experiment_directory",
                description="Analyze an existing experiment output directory to extract comprehensive metrics including consumer metrics, firm metrics, innovation analysis, and macro metrics",
                category=ToolCategory.DATA_ANALYSIS,
                parameters=[
                    ToolParameter("experiment_dir", "str", "Path to the experiment output directory (e.g., '/root/project/agentsociety-ecosim/output/encouraged')", required=True),
                    ToolParameter("experiment_name", "str", "Optional name for the experiment. If not provided, will be derived from directory name", required=False),
                ],
                examples=[
                    "analyze_experiment_directory(\n"
                    "    experiment_dir='/root/project/agentsociety-ecosim/output/encouraged'\n"
                    ")"
                ],
                notes=[
                    "This tool performs comprehensive analysis of an experiment directory",
                    "Returns consumer metrics (expenditure, nutrition, satisfaction, attributes), firm metrics (revenue, profit), innovation metrics (event counts, market share correlations), and macro metrics (GDP)",
                    "Use this tool when analyzing existing experiments to extract all available metrics"
                ]
            )
        )
        
        # 6. compare_experiments
        self.register_tool(
            ToolInfo(
                name="compare_experiments",
                description="对比两个实验的关键指标，生成差异分析和百分比变化",
                category=ToolCategory.DATA_ANALYSIS,
                parameters=[
                    ToolParameter("experiment1_name", "str", "第一个实验名称（通常是 control）", required=True),
                    ToolParameter("experiment2_name", "str", "第二个实验名称（通常是 treatment）", required=True),
                    ToolParameter("metrics_to_compare", "list", "要对比的指标列表（可选，默认对比所有）", required=False),
                    ToolParameter("manifest_path", "str", "manifest.yaml 路径（用于保存对比结果）", required=False),
                ],
                examples=[
                    "compare_experiments(\n"
                    "    experiment1_name='control_policy',\n"
                    "    experiment2_name='treatment_policy',\n"
                    "    manifest_path='experiments/policy_study/manifest.yaml'\n"
                    ")"
                ],
                notes=[
                    "对比结果包括：并排对比、差异分析、百分比变化",
                    "如果提供 manifest_path，对比结果和结论会自动保存到 manifest 中"
                ]
            )
        )
    
    def register_tool(self, tool_info: ToolInfo):
        """注册工具"""
        self.tools[tool_info.name] = tool_info
    
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """获取工具信息"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolInfo]:
        """按分类获取工具"""
        return [tool for tool in self.tools.values() if tool.category == category and tool.enabled]
    
    def get_all_tools(self, enabled_only: bool = True) -> List[ToolInfo]:
        """获取所有工具"""
        tools = list(self.tools.values())
        if enabled_only:
            tools = [tool for tool in tools if tool.enabled]
        return tools
    
    def enable_tool(self, name: str):
        """启用工具"""
        if name in self.tools:
            self.tools[name].enabled = True
    
    def disable_tool(self, name: str):
        """禁用工具"""
        if name in self.tools:
            self.tools[name].enabled = False
    
    def generate_tools_documentation(self) -> str:
        """生成工具文档（用于 system prompt）"""
        lines = ["## Available Tools:\n"]
        
        # 按分类组织工具
        for category in ToolCategory:
            category_tools = self.get_tools_by_category(category)
            if not category_tools:
                continue
            
            lines.append(f"### {category.value}\n")
            
            for tool in category_tools:
                lines.append(f"#### {tool.name}")
                lines.append(f"{tool.description}\n")
                
                # 参数说明
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req_mark = " (required)" if param.required else f" (default: {param.default})" if param.default is not None else " (optional)"
                        lines.append(f"- `{param.name}` ({param.type}): {param.description}{req_mark}")
                    lines.append("")
                
                # 使用示例
                if tool.examples:
                    lines.append("**Example:**")
                    for example in tool.examples:
                        lines.append(f"```python\n{example}\n```")
                    lines.append("")
                
                # 注意事项
                if tool.notes:
                    lines.append("**Notes:**")
                    for note in tool.notes:
                        lines.append(f"- {note}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def generate_tools_summary(self) -> str:
        """生成工具摘要（简短版本）"""
        lines = ["## Available Tools:\n"]
        
        for category in ToolCategory:
            category_tools = self.get_tools_by_category(category)
            if not category_tools:
                continue
            
            lines.append(f"### {category.value}:")
            for tool in category_tools:
                lines.append(f"- **{tool.name}**: {tool.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def extract_from_docstring(self, func) -> Optional[ToolInfo]:
        """从函数 docstring 自动提取工具信息（未来可以扩展）"""
        # 这是一个扩展点，可以从函数的 docstring 和类型注解自动提取工具信息
        # 目前先返回 None，后续可以实现
        return None


# 全局工具管理器实例
_tool_manager = None


def get_tool_manager() -> ToolManager:
    """获取全局工具管理器实例"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager

