from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.message import TextBlock, Msg
from agentscope.tool import Toolkit, ToolResponse
import asyncio
import os
import re
import copy
import json
import subprocess
import threading
from typing import Any, Callable
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager
import yaml
import glob
import functools

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

# Beijing timezone (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))

def _beijing_now() -> datetime:
    """Get current time in Beijing timezone (UTC+8)."""
    return datetime.now(BEIJING_TZ)

def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.expanduser(path))


@contextmanager
def _temp_working_directory(path: str):
    """临时切换工作目录的上下文管理器 - 确保始终恢复"""
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def _temp_env_var(key: str, value: str | None):
    """临时设置环境变量的上下文管理器 - 确保始终恢复"""
    old_value = os.environ.get(key)
    try:
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]


class ToolCallLogger:
    """Thread-safe recorder for tool invocation events."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _now_iso(self) -> str:
        # Use Beijing/Shanghai time for consistency in logs
        return _beijing_now().isoformat()

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            # Normalize timestamp to Beijing time
            event.setdefault("timestamp", self._now_iso())
            self._events.append(event)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


def _tool_response_to_text(response: Any) -> str:
    if isinstance(response, ToolResponse):
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(response)


def _wrap_tool_function(
    func: Callable,
    logger: ToolCallLogger,
) -> Callable:
    """Wrap tool function so that every invocation is recorded."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        event = {
            "tool_name": func.__name__,
            "input": {
                "args": [repr(arg) for arg in args],
                "kwargs": {key: repr(val) for key, val in kwargs.items()},
            },
            # Timestamp will be filled/normalized by logger.record()
        }
        try:
            result = func(*args, **kwargs)
            event["status"] = "success"
            event["output"] = _tool_response_to_text(result)
            return result
        except Exception as exc:
            event["status"] = "error"
            event["error"] = str(exc)
            raise
        finally:
            logger.record(event)

    return wrapper


def _experiment_dir_from_manifest(manifest_path: str) -> str:
    return os.path.dirname(_abspath(manifest_path))


def _get_log_file_from_config(config_path: str) -> str:
    """Generate log file path from config file path: {yaml_basename}.log"""
    config_dir = os.path.dirname(config_path)
    config_basename = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(config_dir, f"{config_basename}.log")


def _load_manifest(manifest_path: str) -> dict:
    manifest_path = _abspath(manifest_path)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _save_manifest(manifest_path: str, data: dict) -> None:
    manifest_path = _abspath(manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def _ensure_manifest_structure(data: dict) -> dict:
    data.setdefault("experiment_info", {})
    data.setdefault("metadata", {})
    metadata = data["metadata"]
    metadata.setdefault("status", "planned")
    metadata.setdefault("runtime", {
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
    })
    data.setdefault("experiment_intervention", {
        "intervention_type": "none",
        "intervention_parameters": {},
    })
    data.setdefault("configurations", {})
    data.setdefault("runs", {})
    data.setdefault("results_summary", {
        "comparison_status": "pending",
        "conclusion": "",
        "insights": [],
    })
    return data


def _init_run_entry(
    config_path: str | None = None,
    parameters_changed: dict | None = None,
    log_file: str | None = None,
) -> dict:
    return {
        "config_path": config_path,
        "parameters_changed": parameters_changed or {},
        "status": "planned",
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "report_file": None,
        "run_log": "",
        "key_metrics": {},
        "log_file": log_file,
        "output_directory": None,
    }


def _ensure_run_log_file(
    manifest: dict,
    manifest_path: str,
    run_label: str,
) -> str:
    """Ensure log_file is set based on config_path"""
    runs = manifest.setdefault("runs", {})
    if run_label not in runs:
        runs[run_label] = _init_run_entry()
    
    run_entry = runs[run_label]
    config_path = run_entry.get("config_path")
    if config_path:
        log_path = _get_log_file_from_config(config_path)
        run_entry["log_file"] = log_path
        return log_path
    return ""


def _normalize_finish_response(self, kwargs):
    """Normalize finish response to avoid errors in block input."""
    msg = kwargs.get("msg")
    if not msg or not isinstance(msg.content, list):
        return kwargs

    for block in msg.content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == self.finish_function_name
        ):
            data = block.get("input")
            if isinstance(data, list):
                block["input"] = {"response": "\n".join(map(str, data))}
            elif isinstance(data, str):
                block["input"] = {"response": data}
            elif data is None:
                block["input"] = {"response": ""}
    return kwargs


def _get_parameter_guide() -> str:
    """
    动态生成参数指南，从 ParameterManager 获取参数元数据
    
    如果 ParameterManager 不可用，则从 YAML 文件读取参数结构作为备用。
    
    Returns:
        格式化的参数指南字符串
    """
    guide_parts = [
        "## Parameter Guide",
        "",
        "Parameters are managed by the ParameterManager. Use the full path format: `category.parameter_name` in YAML files.",
        "",
    ]
    
    # 尝试从 ParameterManager 获取参数
    param_manager_available = False
    try:
        import sys
        from pathlib import Path
        # 获取项目根目录（相对于当前文件）
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # economist/ -> project root
        mcp_server_path = str(project_root / "agentsociety_ecosim" / "mcp_server")
        if mcp_server_path not in sys.path:
            sys.path.insert(0, mcp_server_path)
        
        from parameter_manager import ParameterManager
        
        # 尝试获取已存在的实例
        try:
            param_manager = ParameterManager.get_instance()
            param_manager_available = True
        except (ValueError, RuntimeError):
            # 如果没有实例，尝试创建一个（但可能失败，因为需要 SimulationConfig）
            # 这种情况下我们使用 YAML 备用方案
            pass
        
        if param_manager_available:
            # 获取所有参数
            all_params = param_manager.get_all_parameters(category="all", format="json")
            
            if isinstance(all_params, str):
                import json
                all_params = json.loads(all_params)
            
            # 按类别组织参数
            params_by_category = {}
            for param_name, param_info in all_params.items():
                category = param_info.get("category", "other")
                if category not in params_by_category:
                    params_by_category[category] = []
                params_by_category[category].append((param_name, param_info))
            
            guide_parts.append("### Parameters from ParameterManager")
            guide_parts.append("")
            
            # 优先显示重要类别
            priority_categories = ["tax_policy", "labor_market", "production", "market", "system_scale"]
            other_categories = sorted([cat for cat in params_by_category.keys() if cat not in priority_categories])
            
            for category in priority_categories + other_categories:
                if category not in params_by_category:
                    continue
                
                guide_parts.append(f"#### {category.replace('_', ' ').title()} Parameters")
                guide_parts.append("")
                
                for param_name, param_info in sorted(params_by_category[category]):
                    desc = param_info.get("description", "No description")
                    param_type = param_info.get("type", "unknown")
                    default_val = param_info.get("default_value")
                    min_val = param_info.get("min_value")
                    max_val = param_info.get("max_value")
                    unit = param_info.get("unit", "")
                    
                    # 参数名和类型
                    type_str = param_type
                    if unit:
                        type_str += f" ({unit})"
                    
                    guide_parts.append(f"- **{param_name}** ({type_str})")
                    guide_parts.append(f"  - {desc}")
                    
                    # 范围和默认值
                    info_parts = []
                    if min_val is not None or max_val is not None:
                        range_parts = []
                        if min_val is not None:
                            range_parts.append(f"≥{min_val}")
                        if max_val is not None:
                            range_parts.append(f"≤{max_val}")
                        info_parts.append(f"Range: {', '.join(range_parts)}")
                    
                    if default_val is not None:
                        info_parts.append(f"Default: {default_val}")
                    
                    if info_parts:
                        guide_parts.append(f"  - {', '.join(info_parts)}")
                    
                    guide_parts.append("")
    
    except Exception as e:
        # ParameterManager 不可用，使用 YAML 备用方案
        pass
    
    # 从 YAML 文件读取参数结构（作为补充或备用）
    try:
        # 获取项目根目录（相对于当前文件）
        current_file = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file, ".."))
        default_yaml_path = os.path.join(project_root, "default.yaml")
        if os.path.exists(default_yaml_path):
            with open(default_yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            
            if not param_manager_available:
                guide_parts.append("### Parameters from default.yaml")
                guide_parts.append("")
            
            # 添加创新模块参数（如果 YAML 中有但 ParameterManager 中没有）
            if "innovation" in yaml_data:
                guide_parts.append("#### Innovation Module Parameters")
                guide_parts.append("")
                innovation_params = yaml_data["innovation"]
                for param_name, default_val in innovation_params.items():
                    guide_parts.append(f"- **innovation.{param_name}**")
                    guide_parts.append(f"  - Default: {default_val}")
                    guide_parts.append("")
    
    except Exception as e:
        pass
    
    # 添加使用说明
    guide_parts.extend([
        "### Using Parameters in YAML Files",
        "",
        "When modifying YAML config files, use the dot notation path: `category.parameter_name`",
        "For example:",
        "  - `tax_policy.income_tax_rate`",
        "  - `labor_market.dismissal_rate`",
        "  - `production.labor_productivity_factor`",
        "  - `innovation.policy_encourage_innovation`",
        "",
        "Note: If MCP server is running, you can use MCP tools (get_all_parameters, get_parameter) ",
        "to query parameter metadata dynamically.",
    ])
    
    return "\n".join(guide_parts)


def _build_sys_prompt() -> str:
    """构建系统提示，使用工具管理器生成工具文档"""
    from tool_manager import get_tool_manager
    
    tool_manager = get_tool_manager()
    tools_doc = tool_manager.generate_tools_documentation()
    
    return (
        "You are an Economic System Design Agent specialized in conducting "
        "scientific research through controlled experiments. Your role is to:\n\n"
        "1. **Understand Research Goals**: When given a research question, break it "
        "down into clear, testable hypotheses.\n\n"
        "2. **Design Experiments**: Create controlled experiments by designing "
        "configuration files that test specific hypotheses. You can design experiments "
        "to study various economic phenomena and policy effects.\n\n"
        "3. **Run Experiments**: Execute simulations using the `run_simulation` tool.\n\n"
        "4. **Analyze Results**: Read simulation reports and extract key metrics "
        "using `read_simulation_report`.\n\n"
        "5. **Compare and Conclude**: Use `compare_experiments` to compare control "
        "and treatment groups, then draw scientific conclusions.\n\n"
        f"{tools_doc}\n\n"
        f"{_get_parameter_guide()}\n\n"
        "## Interactive Research Workflow:\n\n"
        "You should engage in a **step-by-step, conversational** workflow with the user. "
        "Do NOT output everything at once. Instead, break down the process into small steps "
        "and wait for user confirmation at each stage.\n\n"
        "**⚠️ CRITICAL: Distinguish Two Scenarios First**\n\n"
        "**Scenario A: User Provides Experiment Directory Path** (e.g., \"analyze /path/to/experiment\")\n"
        "- DO NOT generate hypotheses or ask for confirmation\n"
        "- DIRECTLY call `analyze_experiment_from_manifest(directory_path)`\n"
        "- Be flexible: combine tools for deeper insights based on results\n"
        "- Present findings immediately with tables and actionable recommendations\n\n"
        "**Scenario B: User Has Research Question** (e.g., \"I want to study X\", \"design experiment for Y\")\n"
        "- Follow the full workflow below (hypothesis → design → simulation → analysis)\n\n"
        "**Two Main Modes of Operation:**\n\n"
        "### Mode 1: Design and Run New Experiments (Scenario B)\n\n"
        "**Workflow Steps (Do ONE step at a time, wait for user confirmation):**\n\n"
        "1. **Receive Research Topic**: When the user shares a research topic or question, "
        "you should **first** run an academic search (knowledge base) to ground your hypothesis. Steps:\n"
        "   - Call `query_knowledge_base` with a concise **English-only** query derived from the user's problem (key terms + 1-2 synonyms), "
        "     leave `doc_type` empty (or set `doc_type=\"paper\"`), use `top_k≈10–20`, and a broad year range (e.g., 1900–2025). If 0 results, immediately retry once with a shorter/looser query (still English-only).\n"
        "   - Then present the literature evidence in a **clean, human-readable format**:\n"
        "       - Use a heading like `### Evidence from academic literature`.\n"
        "       - For the **top 3 most relevant papers**, render the title as a Markdown link when `pdf_link` is available, e.g. `[Title (Year, Journal)](URL)`; otherwise show plain text title.\n"
        "       - Under each title, add 2 short bullets:\n"
        "         - `- Core finding:` 1–2 sentences summarizing the main result (based on abstract + introduction).\n"
        "         - `- Takeaway for our question:` 1–2 sentences explaining how this paper informs the current research problem.\n"
        "   - If there are **no useful results** after retries, state clearly that the indexed literature is sparse for this query.\n"
        "   - Then propose a clear, testable **hypothesis** based on the topic and the retrieved context (or lack of context).\n"
        "   - Keep your response concise (2-3 sentences for the hypothesis) and mention if evidence is weak/empty.\n"
        "   - End with: \"Does this hypothesis sound reasonable? Should I proceed with designing the experiment?\"\n"
        "   - **WAIT for user confirmation** before proceeding\n\n"
        "2. **Design Experiment (After Hypothesis Confirmed, WITH RAG-AUGMENTED DESIGN)**: After user confirms the hypothesis, you should:\n"
        "   - **FIRST**: Call `get_available_parameters(category=\"all\")` to discover ALL available parameters\n"
        "     (DO NOT assume you only know innovation parameters - there are many more!).\n"
        "   - **SECOND (RAG for design)**: Use the confirmed hypothesis and candidate policy levers (e.g. tax rate, subsidy, innovation intensity) to call `query_knowledge_base` again with a focused query\n"
        "     (e.g. \"R&D tax credit intensity innovation outcomes\", \"minimum wage labor market experiment design\").\n"
        "     - From the top 3–5 hits, extract how real papers **choose treatment vs control**, typical **treatment intensity ranges**, and which **outcome metrics** they evaluate.\n"
        "     - Summarize these as a short section `### Literature-guided experiment design` with bullets like:\n"
        "       - `- Policy levers used in literature:` …\n"
        "       - `- Typical treatment intensity / parameter ranges:` …\n"
        "       - `- Common outcome metrics:` … (e.g. employment, innovation rate, inequality, GDP).\n"
        "   - Based on the available parameters **and** the RAG evidence, specify the **experiment type**\n"
        "     (controlled experiment with control/treatment groups, or single experiment), and clearly list:\n"
        "       - Which **parameters to adjust** (category + full parameter path from the available list).\n"
        "       - The **control vs treatment values** you propose, with 1-line justification referencing the literature summary where possible.\n"
        "       - The **verification metrics** you'll use and why they match the literature.\n"
        "   - If you need details about a specific parameter, call `get_parameter_info(parameter_name=\"category.param_name\")`.\n"
        "   - Keep this concise (bullet points, 5–12 lines) and clearly separate \"RAG evidence\" vs \"your proposed design\".\n"
        "   - End with: \"Should I proceed to create the configuration files?\".\n"
        "   - **WAIT for user confirmation** before proceeding.\n\n"
        "3. **Initialize Manifest (After Design Confirmed)**: After user confirms the design, you should:\n"
        "   - Use `init_experiment_manifest` to create the manifest\n"
        "   - Confirm the manifest was created successfully\n"
        "   - Tell the user what will be created next (config files)\n"
        "   - **WAIT for user to say \"proceed\", \"continue\", \"yes\", or similar** before creating configs\n\n"
        "4. **Create Config Files (After User Approval)**: Only after explicit user approval:\n"
        "   - **If unsure about parameter names**: Call `get_parameter_info(parameter_name=\"category.param_name\")` to verify\n"
        "   - Use `create_yaml_from_template` to create control and treatment configs\n"
        "   - Make sure you use the correct parameter paths (dot notation, e.g., 'tax_policy.income_tax_rate')\n"
        "   - Briefly summarize what parameters were set in each config (2-3 sentences)\n"
        "   - Tell the user the configs are ready\n"
        "   - End with: \"Configs are ready. Should I start running the simulations?\"\n"
        "   - **WAIT for user confirmation** before running\n\n"
        "5. **Run Simulations (After User Approval)**: Only after explicit user approval:\n"
        "   - Use `run_simulation` to execute the experiments\n"
        "   - Inform the user that simulations are running (this may take time)\n"
        "   - Wait for simulations to complete\n\n"
        "6. **Analyze Results**: After simulations complete:\n"
        "   - Use `read_simulation_report` to extract metrics\n"
        "   - Use `compare_experiments` to compare results\n"
        "   - Present key findings concisely\n\n"
        "### Mode 2: Analyze Existing Experiments\n\n"
        "When the user provides existing experiment directories to analyze, follow this workflow:\n\n"
        "**Workflow Steps for Analysis Mode (Do ONE step at a time, wait for user confirmation):**\n\n"
        "1. **Formulate Hypothesis**: Based on the research question and experiment directories provided:\n"
        "   - Propose a clear, testable **hypothesis** that relates the experimental conditions to expected economic outcomes\n"
        "   - Keep it concise (2-3 sentences)\n"
        "   - End with: \"Does this hypothesis align with your expectations? Should I proceed with the analysis?\"\n"
        "   - **WAIT for user confirmation** before proceeding\n\n"
        "2. **Load Experiment Data**: After hypothesis is confirmed:\n"
        "   - Use `read_simulation_report` with `experiment_name` or `report_file` to load data from each experiment directory\n"
        "   - Identify the experiment directories (usually contain `experiment_manifest.json` or report files)\n"
        "   - Load metrics from all experiments you need to compare\n"
        "   - Confirm the data is loaded successfully\n"
        "   - **WAIT for user confirmation** before proceeding to analysis\n\n"
        "3. **Perform Statistical Analysis**: After data is loaded:\n"
        "   - Analyze consumer metrics: total expenditure, total nutrition value, total satisfaction, total attribute value\n"
        "   - Analyze innovation metrics: innovation frequency, innovation correlation with market share changes\n"
        "   - Analyze firm metrics: total revenue, profit, productivity\n"
        "   - Calculate correlations between innovation events and market share changes (at 1, 2, 3 months after innovation)\n"
        "   - Perform cross-experiment comparisons using `compare_experiments` tool\n"
        "   - Present key statistical findings\n"
        "   - End with: \"Would you like me to proceed with hypothesis verification?\"\n"
        "   - **WAIT for user confirmation** before drawing conclusions\n\n"
        "4. **Verify Hypothesis**: After user confirmation:\n"
        "   - Compare the statistical results across all experiments\n"
        "   - Assess whether the evidence supports or refutes the hypothesis\n"
        "   - Identify which metrics provide the strongest evidence\n"
        "   - Discuss any unexpected findings or limitations\n"
        "   - Present a clear conclusion\n\n"
        "**Key Metrics to Analyze (when analyzing existing experiments):**\n"
        "- **Consumer metrics**: Total expenditure, total nutrition value, total satisfaction value, total attribute value\n"
        "- **Innovation metrics**: Innovation event counts, correlation between innovation frequency and market share changes (at 1, 2, 3 months)\n"
        "- **Firm metrics**: Total revenue, total profit, average profit, productivity factors\n"
        "- **Cross-experiment comparison**: Statistical differences, percentage changes, correlation patterns\n\n"
        "**CRITICAL RULES:**\n"
        "- **ALWAYS call `get_available_parameters()` BEFORE designing experiments** - do NOT assume you know all parameters!\n"
        "- **Use parameter query tools** (`get_available_parameters`, `get_parameter_info`) to discover parameters, especially when designing experiments\n"
        "- **Before proposing any hypothesis**, attempt an academic retrieval: call `query_knowledge_base` with English-only keywords, prefer `doc_type=\"abstract\"`, set `top_k=20`, broad years (e.g., 1950–2025); if 0 results, retry once with a shorter/looser query and report evidence is thin. If a section result looks valuable, call `get_paper_details(paper_id)` to pull the full paper context (abstract + sections) before hypothesizing.\n"
        "- **NEVER** output multiple steps at once (e.g., don't output hypothesis, design, AND config creation all together)\n"
        "- **ALWAYS** wait for explicit user confirmation before moving to the next step\n"
        "- **KEEP** each response concise and focused on ONE thing\n"
        "- **ASK** a clear question at the end of each step to invite user confirmation\n"
        "- **DO NOT** automatically proceed with tool calls without user confirmation\n"
        "- If the user says \"yes\", \"proceed\", \"continue\", \"go ahead\", \"ok\", etc., you can proceed to the next step\n"
        "- If the user provides feedback or asks for changes, address those first before proceeding\n"
        "- **DO NOT limit yourself to innovation parameters** - explore all available parameter categories (tax_policy, production, labor_market, market, etc.)\n\n"
        "## Important Notes:\n\n"
        "- **Be conversational and concise**: Keep responses short and focused. Do not write long paragraphs.\n"
        "- **One step at a time**: Only do ONE thing per response, then wait for user confirmation.\n"
        "- **Use natural language**: Respond like you're having a conversation, not writing a formal report.\n"
        "- Always create config files in a dedicated directory for each research project\n"
        "- Maintain the experiment manifest at every stage so progress/status is transparent\n"
        "- Use descriptive filenames or custom_filename parameter; configs are stored "
        "inside the manifest directory\n"
        "- Wait for simulations to complete before reading reports\n"
        "- Focus on key metrics: employment, income, wealth distribution, Gini coefficient\n"
        "- When comparing, clearly state which is control and which is treatment\n"
        "- Draw conclusions based on statistical differences, not just raw numbers\n\n"
        "**Remember**: Your goal is to have a smooth, back-and-forth conversation with the user, "
        "not to dump all information at once. Each message should be focused on ONE step of the workflow.\n\n"
        "**When analyzing existing experiments (Scenario A):**\n"
        "- **IMPORTANT**: User provides directory path → DIRECTLY analyze, NO hypothesis generation\n"
        "- **Start with**: `analyze_experiment_from_manifest(experiment_dir)` - gets complete overview\n"
        "- **Be flexible**: If initial analysis reveals interesting patterns, dig deeper:\n"
        "  * All failed? → Suggest examining logs, reducing scale\n"
        "  * All succeeded? → Present metrics comparison table, calculate effect sizes\n"
        "  * Mixed results? → Analyze successful ones, diagnose failed ones\n"
        "  * Need details? → Use `read_simulation_report` for specific configs\n"
        "  * Compare designs? → Use `compare_experiments` for statistical tests\n"
        "- **Combine tools creatively** based on what the data shows\n"
        "- **Focus on insights**: Don't just report numbers, explain what they mean\n"
        "- **Key metrics**: employment, income, wealth distribution, Gini coefficient, consumer patterns\n"
        "- **Keep it conversational**: Present findings progressively, not all at once\n\n"
    )


def init_experiment_manifest(
    experiment_dir: str,
    experiment_name: str,
    description: str = "",
    research_question: str = "",
    hypothesis: str = "",
    expected_outcome: str = "",
    intervention_type: str = "none",
    intervention_parameters: dict | None = None,
    configurations: dict | None = None,
    tags: list[str] | None = None,
    manifest_filename: str = "manifest.yaml",
) -> ToolResponse:
    """Create an experiment manifest YAML to track metadata and run statuses."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
    experiments_base_dir = os.path.join(script_dir, "experiments")  # /root/project/agentsociety-ecosim/economist/experiments
    
    # Normalize experiment_dir path
    if os.path.isabs(experiment_dir):
        # Absolute path: check if it's pointing to old location
        if experiment_dir.startswith("/root/project/economist/experiments"):
            # Convert old path to new path
            # Extract the part after /root/project/economist/experiments
            old_base = "/root/project/economist/experiments"
            relative_part = experiment_dir[len(old_base):].lstrip("/")
            if relative_part:
                experiment_dir_abs = os.path.join(experiments_base_dir, relative_part)
            else:
                # Just the base directory, append experiment_name
                experiment_dir_abs = os.path.join(experiments_base_dir, experiment_name)
        else:
            # Other absolute path, use as-is
            experiment_dir_abs = experiment_dir
    else:
        # Relative path: resolve relative to experiments directory
        if experiment_dir.startswith("experiments/"):
            # Remove "experiments/" prefix and resolve
            relative_part = experiment_dir[len("experiments/"):]
            experiment_dir_abs = os.path.join(experiments_base_dir, relative_part)
        elif experiment_dir == "experiments":
            # Just "experiments", use base + experiment_name
            experiment_dir_abs = os.path.join(experiments_base_dir, experiment_name)
        else:
            # Assume it's an experiment name or relative path
            experiment_dir_abs = os.path.join(experiments_base_dir, experiment_dir)
    
    # Ensure experiment_name is included in the final path if not already present
    if experiment_name and experiment_name not in os.path.basename(experiment_dir_abs):
        experiment_dir_abs = os.path.join(experiment_dir_abs, experiment_name)
    
    base_dir = experiment_dir_abs
    
    os.makedirs(experiment_dir_abs, exist_ok=True)
    manifest_path = os.path.join(experiment_dir_abs, manifest_filename)

    created_date = _beijing_now().strftime("%Y-%m-%d")
    data = {
        "experiment_info": {
            "name": experiment_name,
            "description": description,
            "created_date": created_date,
            "author": "DesignAgent",
            "tags": tags or [],
            "notes": "",
            "directory": experiment_dir_abs,
        },
        "metadata": {
            "research_question": research_question,
            "hypothesis": hypothesis,
            "expected_outcome": expected_outcome,
            "status": "planned",
            "runtime": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
            },
        },
        "experiment_intervention": {
            "intervention_type": intervention_type,
            "intervention_parameters": intervention_parameters or {},
        },
        "configurations": {},
        "runs": {},
        "results_summary": {
            "comparison_status": "pending",
            "conclusion": "",
            "insights": [],
        },
    }

    configurations = configurations or {}
    for label, cfg in configurations.items():
        config_path = cfg.get("path")
        params = cfg.get("parameters_changed", {})
        log_file = _get_log_file_from_config(config_path) if config_path else None
        data["configurations"][label] = {
            "path": config_path,
            "parameters_changed": params,
        }
        data["runs"][label] = _init_run_entry(config_path, params, log_file)

    _save_manifest(manifest_path, data)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"<status>success</status>"
                    f"<manifest_path>{manifest_path}</manifest_path>"
                    f"<experiment_dir>{experiment_dir_abs}</experiment_dir>"
                ),
            ),
        ],
    )


def _update_manifest_run(
    manifest_path: str,
    run_label: str,
    status: str,
    report_file: str | None = None,
    run_log: str | None = None,
    output_directory: str | None = None,
) -> dict:
    manifest = _ensure_manifest_structure(_load_manifest(manifest_path))
    runs = manifest.setdefault("runs", {})
    if run_label not in runs:
        runs[run_label] = _init_run_entry()
    run_entry = runs[run_label]
    log_path = _ensure_run_log_file(manifest, manifest_path, run_label)

    now = _beijing_now()
    if output_directory:
        run_entry["output_directory"] = output_directory
    
    if status == "running":
        if manifest["metadata"]["runtime"]["start_time"] is None:
            manifest["metadata"]["runtime"]["start_time"] = now.isoformat()
            manifest["metadata"]["status"] = "running"
        run_entry["status"] = "running"
        run_entry["start_time"] = now.isoformat()
    elif status in {"completed", "failed"}:
        run_entry["status"] = status
        run_entry["end_time"] = now.isoformat()
        if run_entry.get("start_time"):
            start_dt = datetime.fromisoformat(run_entry["start_time"])
            run_entry["duration_seconds"] = (now - start_dt).total_seconds()
        if report_file:
            run_entry["report_file"] = report_file
        if run_log:
            run_entry["run_log"] = run_log

        # Update experiment-level runtime / status if all runs have finished
        if all(entry.get("status") in {"completed", "failed"} for entry in runs.values()):
            manifest["metadata"]["runtime"]["end_time"] = now.isoformat()
            start_time = manifest["metadata"]["runtime"]["start_time"]
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                manifest["metadata"]["runtime"]["duration_seconds"] = (now - start_dt).total_seconds()
            if all(entry.get("status") == "completed" for entry in runs.values()):
                manifest["metadata"]["status"] = "analysis_pending"
            elif any(entry.get("status") == "failed" for entry in runs.values()):
                manifest["metadata"]["status"] = "failed"

    # Update log_file based on config_path if not set
    if not run_entry.get("log_file") and run_entry.get("config_path"):
        run_entry["log_file"] = _get_log_file_from_config(run_entry["config_path"])
    
    _save_manifest(manifest_path, manifest)
    return manifest


def _record_manifest_metrics(
    manifest_path: str,
    run_label: str,
    metrics: dict,
) -> None:
    manifest = _ensure_manifest_structure(_load_manifest(manifest_path))
    runs = manifest.setdefault("runs", {})
    if run_label not in runs:
        runs[run_label] = _init_run_entry()
    runs[run_label]["key_metrics"] = metrics
    _save_manifest(manifest_path, manifest)


def _record_manifest_comparison(
    manifest_path: str,
    comparison: dict,
    conclusion: str = "",
    insights: list[str] | None = None,
) -> None:
    manifest = _ensure_manifest_structure(_load_manifest(manifest_path))
    summary = manifest.setdefault("results_summary", {})
    summary["comparison_status"] = "completed"
    summary["comparison"] = comparison
    summary["conclusion"] = conclusion
    if insights is not None:
        summary["insights"] = insights
    manifest["metadata"]["status"] = "completed"
    _save_manifest(manifest_path, manifest)

def _parse_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    if path == "":
        return tokens

    if not isinstance(path, str):
        raise TypeError("Path must be a string.")

    parts = path.split(".")
    for part in parts:
        if not part:
            raise ValueError(f"Invalid path segment: '{part}' in '{path}'")

        index = 0
        while index < len(part):
            if part[index] == "[":
                match = re.match(r"\[(\d+)\]", part[index:])
                if not match:
                    raise ValueError(
                        f"Invalid list index syntax near '{part[index:]}' "
                        f"in '{path}'",
                    )
                tokens.append(int(match.group(1)))
                index += len(match.group(0))
            else:
                match = re.match(r"[^\[\]]+", part[index:])
                if not match:
                    raise ValueError(
                        f"Invalid path segment near '{part[index:]}' "
                        f"in '{path}'",
                    )
                tokens.append(match.group(0))
                index += len(match.group(0))
    return tokens


def _set_value(
    data: Any,
    tokens: list[Any],
    value: Any,
    create_missing: bool,
) -> Any:
    """Set a value in nested data structure and return the modified data."""
    if not tokens:
        return value
    
    current = data
    for i, token in enumerate(tokens[:-1]):
        next_token = tokens[i + 1]
        if isinstance(token, int):
            if not isinstance(current, list):
                raise TypeError(
                    f"Expected list at segment '{token}', "
                    f"but found type {type(current).__name__}",
                )
            while len(current) <= token:
                if not create_missing:
                    raise KeyError(
                        f"Index {token} out of range and create_missing=False",
                    )
                current.append({} if isinstance(next_token, str) else [])
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Expected mapping at segment '{token}', "
                    f"but found type {type(current).__name__}",
                )
            if token not in current or current[token] is None:
                if not create_missing:
                    raise KeyError(
                        f"Key '{token}' not found and create_missing=False",
                    )
                current[token] = {} if isinstance(next_token, str) else []
            current = current[token]

    last_token = tokens[-1]
    if isinstance(last_token, int):
        if not isinstance(current, list):
            raise TypeError(
                f"Expected list at terminal segment '{last_token}', "
                f"but found type {type(current).__name__}",
            )
        while len(current) <= last_token:
            if not create_missing:
                raise KeyError(
                    f"Index {last_token} out of range and create_missing=False",
                )
            current.append(None)
        current[last_token] = value
    else:
        if not isinstance(current, dict):
            raise TypeError(
                f"Expected mapping at terminal segment '{last_token}', "
                f"but found type {type(current).__name__}",
            )
        current[last_token] = value
    
    return data


def _delete_value(
    data: Any,
    tokens: list[Any],
) -> tuple[Any, bool]:
    current = data
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(current, list):
                return data, False
            if token >= len(current):
                return data, False
            current = current[token]
        else:
            if not isinstance(current, dict):
                return data, False
            if token not in current:
                return data, False
            current = current[token]

    last_token = tokens[-1]
    if isinstance(last_token, int):
        if not isinstance(current, list):
            return data, False
        if last_token >= len(current):
            return data, False
        current.pop(last_token)
        return data, True

    if not isinstance(current, dict):
        return data, False
    if last_token not in current:
        return data, False
    del current[last_token]
    return data, True


def _generate_unique_filename(
    base_dir: str,
    param_changes: list[tuple[str, Any]],
    prefix: str = "run",
) -> str:
    """Generate a unique filename based on parameter changes."""
    name_parts = [prefix]
    for path, value in param_changes:
        path_parts = path.split(".") if path else []
        param_name = path_parts[-1] if path_parts else "root"
        
        if isinstance(value, bool):
            value_str = str(value)
        elif isinstance(value, (int, float)):
            value_str = str(value).replace(".", "_")
        elif isinstance(value, str):
            value_str = value[:10].replace(" ", "_").replace("/", "_")
        else:
            value_str = str(value)[:10].replace(" ", "_")
        
        param_name = re.sub(r'[^\w-]', '_', param_name)
        value_str = re.sub(r'[^\w-]', '_', str(value_str))
        
        name_parts.append(f"{param_name}_{value_str}")
    
    if len(name_parts) > 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
    else:
        base_name = "_".join(name_parts)
    
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_name}.yaml"
        else:
            filename = f"{base_name}_{counter}.yaml"
        
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            return filename
        
        counter += 1
        if counter > 1000:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            return f"{prefix}_{timestamp}.yaml"


def _extract_scalar_value(node: Any, target_key: str) -> Any:
    """Recursively extract the first non-container value for the given key."""
    if isinstance(node, dict):
        if target_key in node and not isinstance(node[target_key], (dict, list)):
            return node[target_key]
        for value in node.values():
            result = _extract_scalar_value(value, target_key)
            if result is not None:
                return result
    elif isinstance(node, list):
        for item in node:
            result = _extract_scalar_value(item, target_key)
            if result is not None:
                return result
    return None


def _ensure_experiment_metadata(data: dict) -> tuple[str, str]:
    """Ensure experiment_name and experiment_output_dir are present in config data."""
    experiment_name = data.get("experiment_name")
    experiment_output_dir = data.get("experiment_output_dir")
    
    if experiment_name and experiment_output_dir:
        return experiment_name, experiment_output_dir
    
    num_households = _extract_scalar_value(data.get("system_scale", {}), "num_households")
    if num_households is None:
        num_households = _extract_scalar_value(data, "num_households") or 0
    num_iterations = _extract_scalar_value(data.get("system_scale", {}), "num_iterations")
    if num_iterations is None:
        num_iterations = _extract_scalar_value(data, "num_iterations") or 0
    
    try:
        num_households = int(num_households)
    except (TypeError, ValueError):
        num_households = 0
    try:
        num_iterations = int(num_iterations)
    except (TypeError, ValueError):
        num_iterations = 0
    
    timestamp = _beijing_now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_name or f"exp_{num_households}h_{num_iterations}m_{timestamp}"
    # 获取项目根目录（相对于当前文件）
    current_file = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file, ".."))
    base_output_dir = os.path.join(project_root, "output")
    experiment_output_dir = experiment_output_dir or os.path.join(base_output_dir, experiment_name)
    
    data["experiment_name"] = experiment_name
    data["experiment_output_dir"] = experiment_output_dir
    return experiment_name, experiment_output_dir


def _check_path_exists(data: Any, tokens: list[Any]) -> bool:
    """Check if a path exists in the data structure."""
    if not tokens:
        return True
    
    current = data
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return False
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                return False
            current = current[token]
    return True


def create_yaml_from_template(
    source_file: str = "default.yaml",
    output_dir: str | None = None,
    parameter_changes: dict[str, Any] | None = None,
    custom_filename: str | None = None,
    allow_new_parameters: bool = False,
    manifest_path: str | None = None,
    config_label: str | None = None,
) -> ToolResponse:
    """Create a new YAML file by modifying parameters from a source template.
    
    IMPORTANT: By default, this tool ONLY modifies existing parameters. It will
    NOT create new parameters.
    """
    if os.path.isabs(source_file):
        source_path = source_file
    else:
        source_path = os.path.abspath(os.path.expanduser(source_file))
    manifest_dir = None
    if manifest_path:
        manifest_dir = os.path.dirname(_abspath(manifest_path))
    
    if not os.path.exists(source_path):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Source file not found: {source_path}</error>"
                    ),
                ),
            ],
        )
    
    if output_dir is None:
        if manifest_dir:
            output_dir = manifest_dir
        else:
            output_dir = os.path.dirname(source_path) or "."
    else:
        if manifest_dir and not os.path.isabs(output_dir):
            output_dir = os.path.abspath(os.path.join(manifest_dir, output_dir))
        else:
            output_dir = (
                output_dir
                if os.path.isabs(output_dir)
                else os.path.abspath(os.path.expanduser(output_dir))
            )
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
        data = yaml.safe_load(content) or {}
    except Exception as exc:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to read source file: {exc}</error>"
                    ),
                ),
            ],
        )
    
    parameter_changes = parameter_changes or {}
    changes_applied: list[tuple[str, Any, Any]] = []
    errors: list[str] = []
    
    data = copy.deepcopy(data)
    
    for path, new_value in parameter_changes.items():
        if not isinstance(path, str) or not path.strip():
            errors.append(f"Invalid path: {repr(path)}")
            continue
        
        tokens = _parse_path(path)
        
        if not allow_new_parameters:
            if not _check_path_exists(data, tokens):
                errors.append(
                    f"Parameter '{path}' does not exist in source file. "
                    f"You can only modify existing parameters."
                )
                continue
        
        old_value = data
        try:
            for token in tokens:
                if isinstance(token, int):
                    if not isinstance(old_value, list) or token >= len(old_value):
                        old_value = None
                        break
                    old_value = old_value[token]
                else:
                    if not isinstance(old_value, dict) or token not in old_value:
                        old_value = None
                        break
                    old_value = old_value[token]
        except (TypeError, KeyError, IndexError):
            old_value = None
        
        try:
            if len(tokens) == 0:
                data = new_value
            else:
                data = _set_value(data, tokens, new_value, create_missing=allow_new_parameters)
            changes_applied.append((path, old_value, new_value))
        except Exception as e:
            errors.append(f"Failed to set {path}: {e}")
    
    if data is None:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Resulting YAML data is None.</error>"
                    ),
                ),
            ],
        )
    
    experiment_name_value, experiment_output_dir = _ensure_experiment_metadata(data)
    
    if custom_filename:
        safe_name = re.sub(r'[^\w.-]', '_', custom_filename)
        if not safe_name.endswith('.yaml') and not safe_name.endswith('.yml'):
            safe_name += '.yaml'
        
        counter = 0
        base_name = safe_name.rsplit('.', 1)[0]
        ext = safe_name.rsplit('.', 1)[1] if '.' in safe_name else 'yaml'
        while True:
            if counter == 0:
                filename = f"{base_name}.{ext}"
            else:
                filename = f"{base_name}_{counter}.{ext}"
            
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                break
            
            counter += 1
            if counter > 1000:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{base_name}_{timestamp}.{ext}"
                break
    else:
        param_changes_list = [(path, new_val) for path, old_val, new_val in changes_applied]
        filename = _generate_unique_filename(output_dir, param_changes_list)
    
    output_path = os.path.join(output_dir, filename)
    try:
        yaml_dump = yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        if not yaml_dump or yaml_dump.strip() in ("null", "~", ""):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Resulting YAML content is empty or null.</error>"
                        ),
                    ),
                ],
            )
        
        if not yaml_dump.endswith("\n"):
            yaml_dump += "\n"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_dump)
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to write output file: {e}</error>"
                    ),
                ),
            ],
        )
    
    changes_summary = []
    for path, old_val, new_val in changes_applied:
        changes_summary.append(f"  {path}: {repr(old_val)} -> {repr(new_val)}")
    
    response_parts = [
        f"<status>success</status>",
        f"<source_file>{source_path}</source_file>",
        f"<output_file>{output_path}</output_file>",
        f"<filename>{filename}</filename>",
        f"<experiment_name>{experiment_name_value}</experiment_name>",
        f"<experiment_output_dir>{experiment_output_dir}</experiment_output_dir>",
    ]
    
    if changes_applied:
        response_parts.append(f"<changes>\n" + "\n".join(changes_summary) + "\n</changes>")
    
    if errors:
        response_parts.append(f"<errors>\n" + "\n".join(f"  {e}" for e in errors) + "\n</errors>")

    if manifest_path and config_label:
        manifest = _ensure_manifest_structure(_load_manifest(manifest_path))
        manifest["configurations"][config_label] = {
            "path": output_path,
            "parameters_changed": {path: new for path, _, new in changes_applied},
            "experiment_name": experiment_name_value,
            "experiment_output_dir": experiment_output_dir,
        }
        runs = manifest.setdefault("runs", {})
        log_file = _get_log_file_from_config(output_path)
        if config_label not in runs:
            runs[config_label] = _init_run_entry(
                output_path,
                manifest["configurations"][config_label]["parameters_changed"],
                log_file=log_file,
            )
        else:
            runs[config_label]["config_path"] = output_path
            runs[config_label]["parameters_changed"] = manifest["configurations"][config_label]["parameters_changed"]
            runs[config_label]["log_file"] = log_file
        runs[config_label]["output_directory"] = experiment_output_dir
        _save_manifest(manifest_path, manifest)
        response_parts.append(f"<manifest_updated>{manifest_path}</manifest_updated>")
    
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text="\n".join(response_parts),
            ),
        ],
    )


def run_simulation(
    config_file: str,
    experiment_name: str | None = None,
    timeout: int = 3600,
    manifest_path: str | None = None,
    run_label: str | None = None,
) -> ToolResponse:
    """Run an economic simulation experiment using the specified YAML config file.
    
    This tool executes the simulation script and waits for completion. The simulation
    will generate a report in the output directory.

    Args:
        config_file:
            Path to the YAML configuration file to use for the simulation.
            Can be absolute or relative path.
        experiment_name:
            Optional name for the experiment. If not provided, will be derived
            from the config filename.
        timeout:
            Maximum time in seconds to wait for simulation to complete.
            Default is 3600 seconds (1 hour).

    Returns:
        ToolResponse:
            A response containing the experiment name, output directory, and
            status of the simulation run.
    """
    if os.path.isabs(config_file):
        config_path = config_file
    else:
        config_path = os.path.abspath(os.path.expanduser(config_file))

    if not os.path.exists(config_path):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Config file not found: {config_path}</error>"
                    ),
                ),
            ],
        )
    
    # Determine experiment name
    config_data = {}
    try:
        with open(config_path, "r", encoding="utf-8") as config_f:
            config_data = yaml.safe_load(config_f) or {}
    except Exception:
        config_data = {}
    
    config_experiment_name = _extract_scalar_value(config_data, "experiment_name")
    config_output_dir = _extract_scalar_value(config_data, "experiment_output_dir")
    
    if experiment_name is None:
        experiment_name = os.path.splitext(os.path.basename(config_path))[0]
    if config_experiment_name:
        experiment_name = config_experiment_name
    
    if config_output_dir:
        experiment_output_dir = config_output_dir
    else:
        # Use relative path: agentsociety-ecosim is now the parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
        ecosim_root = os.path.abspath(os.path.join(current_dir, ".."))  # /root/project/agentsociety-ecosim
        output_base = os.path.join(ecosim_root, "output")
        experiment_output_dir = os.path.join(output_base, experiment_name)
    
    # Path to simulation wrapper script
    current_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
    sim_script = os.path.join(current_dir, "run_simulation.sh")
    if not os.path.exists(sim_script):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Simulation script not found: {sim_script}</error>"
                    ),
                ),
            ],
        )
    
    # Run simulation
    # 在更新为 running 之前，先完成所有必要的检查和准备
    try:
        print(f"🚀 Starting simulation with config: {config_path}")
        print(f"📁 Experiment name: {experiment_name}")

        # 提前更新 log_file（在运行前）
        if manifest_path and run_label:
            try:
                manifest = _ensure_manifest_structure(_load_manifest(manifest_path))
                runs = manifest.setdefault("runs", {})
                if run_label in runs:
                    runs[run_label]["log_file"] = _get_log_file_from_config(config_path)
                    _save_manifest(manifest_path, manifest)
            except Exception as e:
                # 如果更新 manifest 失败，记录但不阻止运行
                print(f"Warning: Failed to update manifest log_file: {e}")
        
        # 更新为 running 状态（在 subprocess.run 之前）
        if manifest_path and run_label:
            try:
                _update_manifest_run(manifest_path, run_label, "running", output_directory=experiment_output_dir)
            except Exception as e:
                # 如果更新状态失败，记录但不阻止运行
                print(f"Warning: Failed to update manifest status to running: {e}")
        
        # Note: run_simulation.sh automatically creates {yaml_basename}.log in the same directory as the yaml file
        try:
            # 获取项目根目录（相对于当前文件）
            current_file = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file, ".."))
            
            result = subprocess.run(
                [sim_script, config_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=project_root,
            )
        except subprocess.TimeoutExpired as e:
            # 超时错误：立即更新为 failed
            if manifest_path and run_label:
                try:
                    _update_manifest_run(
                        manifest_path,
                        run_label,
                        "failed",
                        run_log=f"Timeout after {timeout} seconds: {str(e)}",
                        output_directory=experiment_output_dir,
                    )
                except Exception as update_err:
                    print(f"Error updating manifest after timeout: {update_err}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Simulation timed out after {timeout} seconds</error>"
                        ),
                    ),
                ],
            )
        except Exception as e:
            # subprocess.run 执行前的任何错误：立即更新为 failed
            if manifest_path and run_label:
                try:
                    _update_manifest_run(
                        manifest_path,
                        run_label,
                        "failed",
                        run_log=f"Failed to start simulation: {str(e)}",
                        output_directory=experiment_output_dir,
                    )
                except Exception as update_err:
                    print(f"Error updating manifest after subprocess error: {update_err}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Failed to start simulation: {e}</error>"
                        ),
                    ),
                ],
            )
        
        # 检查 subprocess 返回码
        if result.returncode != 0:
            # 非零返回码：立即更新为 failed
            if manifest_path and run_label:
                try:
                    _update_manifest_run(
                        manifest_path,
                        run_label,
                        "failed",
                        run_log=result.stderr[-2000:] if result.stderr else f"Return code: {result.returncode}",
                        output_directory=experiment_output_dir,
                    )
                except Exception as update_err:
                    print(f"Error updating manifest after failure: {update_err}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Simulation failed with return code {result.returncode}</error>"
                            f"<stdout>{result.stdout}</stdout>"
                            f"<stderr>{result.stderr}</stderr>"
                        ),
                    ),
                ],
            )
        
        # 处理成功情况
        try:
            # Find the output directory (respecting overrides if provided in YAML)
            # Find the most recent report file
            report_files = glob.glob(
                os.path.join(experiment_output_dir, "simulation_report_*.json")
            )
            
            latest_report = None
            if report_files:
                latest_report = max(report_files, key=os.path.getmtime)
            
            tail_stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
            response_parts = [
                f"<status>success</status>",
                f"<experiment_name>{experiment_name}</experiment_name>",
                f"<config_file>{config_path}</config_file>",
                f"<output_directory>{experiment_output_dir}</output_directory>",
            ]
            
            if latest_report:
                response_parts.append(f"<report_file>{latest_report}</report_file>")
            
            response_parts.append(
                f"<stdout_summary>{tail_stdout}</stdout_summary>"
            )

            if manifest_path and run_label:
                try:
                    _update_manifest_run(
                        manifest_path,
                        run_label,
                        "completed",
                        report_file=latest_report,
                        run_log=tail_stdout,
                        output_directory=experiment_output_dir,
                    )
                except Exception as update_err:
                    print(f"Error updating manifest after completion: {update_err}")
            
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="\n".join(response_parts),
                    ),
                ],
            )
        except Exception as e:
            # 处理成功后的任何错误（如查找 report 文件失败）：更新为 failed
            if manifest_path and run_label:
                try:
                    _update_manifest_run(
                        manifest_path,
                        run_label,
                        "failed",
                        run_log=f"Error processing simulation results: {str(e)}",
                        output_directory=experiment_output_dir,
                    )
                except Exception as update_err:
                    print(f"Error updating manifest after processing error: {update_err}")
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Simulation completed but failed to process results: {e}</error>"
                        ),
                    ),
                ],
            )
        
    except Exception as e:
        # 最外层异常捕获：确保任何未预料的错误都更新状态
        if manifest_path and run_label:
            try:
                _update_manifest_run(
                    manifest_path,
                    run_label,
                    "failed",
                    run_log=f"Unexpected error: {str(e)}",
                    output_directory=experiment_output_dir,
                )
            except Exception as update_err:
                print(f"Error updating manifest after unexpected error: {update_err}")
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to run simulation: {e}</error>"
                    ),
                ),
            ],
        )


def read_simulation_report(
    report_file: str | None = None,
    experiment_name: str | None = None,
    manifest_path: str | None = None,
    run_label: str | None = None,
) -> ToolResponse:
    """Read and extract key metrics from a simulation report.
    
    This tool reads the JSON report file generated by a simulation and extracts
    the most important economic indicators for analysis.
    
    Args:
        report_file:
            Direct path to the report JSON file. If provided, this takes precedence.
        experiment_name:
            Name of the experiment. If report_file is not provided, will look for
            the most recent report in output/{experiment_name}/.
    
    Returns:
        ToolResponse:
            A response containing extracted key metrics and summary statistics.
    """
    if report_file:
        if os.path.isabs(report_file):
            report_path = report_file
        else:
            report_path = os.path.abspath(os.path.expanduser(report_file))
    elif experiment_name:
        # 获取项目根目录（相对于当前文件）
        current_file = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file, ".."))
        output_dir = os.path.join(project_root, "output", experiment_name)
        report_files = glob.glob(
            os.path.join(output_dir, "simulation_report_*.json")
        )
        if not report_files:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>No report files found in {output_dir}</error>"
                        ),
                    ),
                ],
            )
        report_path = max(report_files, key=os.path.getmtime)
    else:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Either report_file or experiment_name must be provided</error>"
                    ),
                ),
            ],
        )
    
    if not os.path.exists(report_path):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Report file not found: {report_path}</error>"
                    ),
                ),
            ],
        )
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Extract key metrics
        key_metrics = {}
        
        # Simulation summary
        if "simulation_summary" in report:
            summary = report["simulation_summary"]
            key_metrics["simulation"] = {
                "iterations": summary.get("total_iterations", 0),
                "households": summary.get("total_households", 0),
                "firms": summary.get("total_firms", 0),
            }
        
        # Latest economic indicators (from last month)
        if "economic_indicators" in report and report["economic_indicators"]:
            indicators = report["economic_indicators"]
            if isinstance(indicators, list) and len(indicators) > 0:
                latest = indicators[-1]
                
                # Employment statistics
                if "employment_statistics" in latest:
                    emp = latest["employment_statistics"]
                    key_metrics["employment"] = {
                        "labor_utilization_rate": emp.get("labor_utilization_rate", 0),
                        "household_unemployment_rate": emp.get("household_unemployment_rate", 0),
                        "total_employed": emp.get("total_labor_force_employed", 0),
                    }
                
                # Income and expenditure
                if "income_expenditure_analysis" in latest:
                    income_exp = latest["income_expenditure_analysis"]
                    key_metrics["income_expenditure"] = {
                        "average_monthly_income": income_exp.get("average_monthly_income", 0),
                        "average_monthly_expenditure": income_exp.get("average_monthly_expenditure", 0),
                        "monthly_savings_rate": income_exp.get("monthly_savings_rate", 0),
                    }
                
                # Wealth distribution
                if "wealth_distribution" in latest:
                    wealth = latest["wealth_distribution"]
                    key_metrics["wealth"] = {
                        "average_wealth": wealth.get("average_wealth", 0),
                        "median_wealth": wealth.get("median_wealth", 0),
                        "gini_coefficient": wealth.get("gini_coefficient", 0),
                    }
        
        # Economic trends summary
        if "economic_trends" in report and "trend_summary" in report["economic_trends"]:
            trends = report["economic_trends"]["trend_summary"]
            key_metrics["trends"] = {}
            for trend_name, trend_info in trends.items():
                if isinstance(trend_info, dict):
                    key_metrics["trends"][trend_name] = {
                        "direction": trend_info.get("direction", "unknown"),
                        "change_rate": trend_info.get("change_rate", 0),
                        "start_value": trend_info.get("start_value", 0),
                        "end_value": trend_info.get("end_value", 0),
                    }
        
        response_parts = [
            f"<status>success</status>",
            f"<report_file>{report_path}</report_file>",
            f"<key_metrics>{json.dumps(key_metrics, indent=2, ensure_ascii=False)}</key_metrics>",
        ]

        if manifest_path and run_label:
            _record_manifest_metrics(manifest_path, run_label, key_metrics)
            response_parts.append(f"<manifest_updated>{manifest_path}</manifest_updated>")
        
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="\n".join(response_parts),
                ),
            ],
        )
        
    except json.JSONDecodeError as e:
        return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                        f"<status>error</status>"
                        f"<error>Failed to parse JSON report: {e}</error>"
                    ),
                ),
            ],
        )
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to read report: {e}</error>"
                ),
            ),
        ],
    )


def get_available_parameters(
    category: str = "all",
) -> ToolResponse:
    """Get all available parameters in the simulation system, organized by category.
    
    This tool should be called BEFORE designing experiments to understand what parameters
    are available to modify. Use this to discover parameters beyond just innovation-related ones.
    
    Args:
        category:
            Parameter category filter. Options:
            - "all" (default): Get all parameters
            - "tax_policy": Tax-related parameters
            - "production": Production parameters
            - "labor_market": Labor market parameters
            - "market": Market parameters
            - "system_scale": System scale parameters
            - "redistribution": Redistribution parameters
            - "performance": Performance parameters
            - "monitoring": Monitoring parameters
            - "innovation": Innovation parameters
    
    Returns:
        ToolResponse:
            A response containing all available parameters organized by category.
    """
    try:
        import sys
        
        # Add the MCP server directory to path
        mcp_server_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 
                        "../agentsociety-ecosim/agentsociety_ecosim/mcp_server")
        )
        if mcp_server_dir not in sys.path:
            sys.path.insert(0, mcp_server_dir)
        
        # Try to use ParameterManager
        try:
            from parameter_manager import ParameterManager
            
            # Try to get existing instance
            param_manager = None
            try:
                param_manager = ParameterManager.get_instance()
            except (ValueError, RuntimeError, FileNotFoundError, Exception) as e:
                # Try to create a new instance with config
                try:
                    from agentsociety_ecosim.simulation.joint_debug_test import SimulationConfig
                    # Set correct working directory for data files
                    # Get the agentsociety-ecosim project root directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
                    ecosim_root = os.path.abspath(os.path.join(current_dir, ".."))  # /root/project/agentsociety-ecosim
                    
                    # 使用上下文管理器安全地切换目录和环境变量
                    with _temp_working_directory(ecosim_root), _temp_env_var('MCP_MODE', '1'):
                        config = SimulationConfig()
                        ParameterManager.reset_instance()
                        param_manager = ParameterManager.get_instance(config=config)
                except Exception as config_error:
                    # If config creation fails, try to get metadata without config
                    param_manager = None
            
            if param_manager is None:
                raise ImportError("Cannot initialize ParameterManager")
            
            # Get parameters
            import json  # Import json at the beginning of this block
            all_params = param_manager.get_all_parameters(category=category, format="json")
            
            if isinstance(all_params, str):
                try:
                    params_data = json.loads(all_params)
                except:
                    params_data = {"raw": all_params}
            else:
                params_data = all_params
            
            response_text = f"<status>success</status>\n"
            response_text += f"<category>{category}</category>\n"
            response_text += f"<parameters>\n{json.dumps(params_data, indent=2, ensure_ascii=False)}\n</parameters>\n"
            response_text += f"\n**Total parameters found: {len(params_data.get('parameters', {})) if isinstance(params_data, dict) else 'N/A'}**\n"
            response_text += f"\nUse this information to identify which parameters you want to adjust for your experiment."
            
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=response_text,
                    ),
                ],
            )
        except (ImportError, FileNotFoundError, Exception) as e:
            # Fallback: read from YAML
            import json  # Import json for YAML fallback
            default_yaml_path = os.path.join(os.path.dirname(__file__), "default.yaml")
            if os.path.exists(default_yaml_path):
                import yaml
                with open(default_yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
                
                response_text = f"<status>success</status>\n"
                response_text += f"<category>{category}</category>\n"
                response_text += f"<note>Using YAML fallback (ParameterManager not available)</note>\n"
                response_text += f"<parameters>\n{json.dumps(yaml_data, indent=2, ensure_ascii=False)}\n</parameters>\n"
                
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=response_text,
                        ),
                    ],
                )
            else:
                raise ValueError(f"Cannot access ParameterManager or YAML file: {e}")
                
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to get parameters: {e}</error>"
                    ),
                ),
            ],
        )


def get_parameter_info(
    parameter_name: str,
) -> ToolResponse:
    """Get detailed information about a specific parameter.
    
    Use this tool to get details about a specific parameter, including its current value,
    allowed range, type, and description.
    
    Args:
        parameter_name:
            The parameter name in dot notation format (e.g., 'innovation.policy_encourage_innovation'
            or 'tax_policy.income_tax_rate')
    
    Returns:
        ToolResponse:
            A response containing detailed parameter information.
    """
    try:
        import sys
        
        # Add the MCP server directory to path
        mcp_server_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 
                        "../agentsociety-ecosim/agentsociety_ecosim/mcp_server")
        )
        if mcp_server_dir not in sys.path:
            sys.path.insert(0, mcp_server_dir)
        
        # Try to use ParameterManager
        try:
            from parameter_manager import ParameterManager
            
            # Try to get existing instance
            param_manager = None
            try:
                param_manager = ParameterManager.get_instance()
            except (ValueError, RuntimeError, FileNotFoundError, Exception) as e:
                # Try to create a new instance with config
                try:
                    from agentsociety_ecosim.simulation.joint_debug_test import SimulationConfig
                    # Set correct working directory for data files
                    # Get the agentsociety-ecosim project root directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
                    ecosim_root = os.path.abspath(os.path.join(current_dir, ".."))  # /root/project/agentsociety-ecosim
                    
                    # 使用上下文管理器安全地切换目录和环境变量
                    with _temp_working_directory(ecosim_root), _temp_env_var('MCP_MODE', '1'):
                        config = SimulationConfig()
                        ParameterManager.reset_instance()
                        param_manager = ParameterManager.get_instance(config=config)
                except Exception as config_error:
                    # If config creation fails, try to get metadata without config
                    param_manager = None
            
            if param_manager is None:
                raise ImportError("Cannot initialize ParameterManager")
            
            # Get parameter info
            param_info = param_manager.get_parameter(parameter_name)
            
            import json
            if isinstance(param_info, str):
                try:
                    param_data = json.loads(param_info)
                except:
                    param_data = {"raw": param_info}
            else:
                param_data = param_info
            
            response_text = f"<status>success</status>\n"
            response_text += f"<parameter_name>{parameter_name}</parameter_name>\n"
            response_text += f"<parameter_info>\n{json.dumps(param_data, indent=2, ensure_ascii=False)}\n</parameter_info>\n"
            
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=response_text,
                    ),
                ],
            )
        except (ImportError, FileNotFoundError, Exception) as e:
            # Fallback: try to get parameter info from YAML file
            try:
                import yaml
                default_yaml_path = os.path.join(os.path.dirname(__file__), "default.yaml")
                if os.path.exists(default_yaml_path):
                    with open(default_yaml_path, "r", encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f) or {}
                    
                    # Parse parameter path (e.g., "production.labor_productivity_factor")
                    path_parts = parameter_name.split(".")
                    value = yaml_data
                    for part in path_parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            raise ValueError(f"Parameter '{parameter_name}' not found in YAML")
                    
                    import json
                    param_data = {
                        "name": parameter_name,
                        "value": value,
                        "type": type(value).__name__,
                        "note": "Information retrieved from YAML file (ParameterManager not available)"
                    }
                    
                    response_text = f"<status>success</status>\n"
                    response_text += f"<parameter_name>{parameter_name}</parameter_name>\n"
                    response_text += f"<parameter_info>\n{json.dumps(param_data, indent=2, ensure_ascii=False)}\n</parameter_info>\n"
                    response_text += f"\n<note>ParameterManager not available, using YAML fallback</note>\n"
                    
                    return ToolResponse(
                        content=[
                            TextBlock(
                                type="text",
                                text=response_text,
                            ),
                        ],
                    )
            except Exception as yaml_error:
                pass
            
            # If all else fails, return error
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Failed to get parameter info for '{parameter_name}': {str(e)}</error>"
                            f"<note>ParameterManager initialization failed. You can still try to use the parameter "
                            f"in YAML files using dot notation: {parameter_name}</note>"
                        ),
                    ),
                ],
            )
                
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to get parameter info for '{parameter_name}': {e}</error>"
                    ),
                ),
            ],
        )


def analyze_experiment_from_manifest(
    experiment_dir: str,
    include_output_analysis: bool = True,
) -> ToolResponse:
    """🎯 一键完整分析实验 - 用户要求\"读取/分析实验目录\"时优先使用此工具！
    
    这是最推荐的实验分析工具，可以一次性完成所有分析任务：
    
    ✅ 自动读取 manifest.yaml（实验设计、假设、配置）
    ✅ 自动定位 output 目录（无需手动指定）
    ✅ 自动提取关键指标（就业、收入、财富、基尼系数等）
    ✅ 自动对比配置组（control vs treatment）
    ✅ 生成结构化分析结果（JSON格式，易于解析展示）
    
    适用场景：
    - 用户提供实验项目目录路径（包含manifest.yaml）
    - 需要完整分析（设计+结果+对比）
    - 实验已完成运行
    
    Args:
        experiment_dir: 实验项目目录路径（包含manifest.yaml的目录）
        include_output_analysis: 是否包含详细结果分析（默认True，强烈推荐）
    
    Returns:
        ToolResponse: 包含实验信息、配置对比、关键指标、运行状态的完整分析
    """
    try:
        # 确保路径存在
        experiment_dir_abs = os.path.abspath(experiment_dir)
        if not os.path.exists(experiment_dir_abs):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>实验目录不存在: {experiment_dir_abs}</error>"
                        ),
                    ),
                ],
            )
        
        # 读取manifest
        manifest_path = os.path.join(experiment_dir_abs, "manifest.yaml")
        if not os.path.exists(manifest_path):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>未找到manifest.yaml: {manifest_path}</error>"
                        ),
                    ),
                ],
            )
        
        manifest = _load_manifest(manifest_path)
        if not manifest:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>manifest文件为空或格式错误</error>"
                        ),
                    ),
                ],
            )
        
        # 提取基本信息
        exp_info = manifest.get("experiment_info", {})
        metadata = manifest.get("metadata", {})
        configurations = manifest.get("configurations", {})
        runs = manifest.get("runs", {})
        
        # 构建分析结果
        analysis = {
            "实验基本信息": {
                "名称": exp_info.get("name", "未命名"),
                "描述": exp_info.get("description", "无"),
                "研究问题": metadata.get("research_question", "未定义"),
                "假设": metadata.get("hypothesis", "未定义"),
                "状态": metadata.get("status", "unknown"),
            },
            "配置组": {},
            "运行状态": {},
            "详细结果": {},
        }
        
        # 统计信息
        total_configs = len(configurations)
        completed_runs = sum(1 for r in runs.values() if r.get("status") == "completed")
        failed_runs = sum(1 for r in runs.values() if r.get("status") == "failed")
        
        # 处理每个配置组
        for config_name, config_detail in configurations.items():
            # 配置信息
            analysis["配置组"][config_name] = {
                "参数变更": config_detail.get("parameters_changed", {}),
                "输出目录": config_detail.get("experiment_output_dir", "未定义"),
            }
            
            # 运行状态
            run_detail = runs.get(config_name, {})
            status = run_detail.get("status", "unknown")
            analysis["运行状态"][config_name] = {
                "状态": status,
                "报告文件": run_detail.get("report_file"),
            }
            
            # 如果成功且需要详细分析
            if include_output_analysis and status == "completed":
                output_dir = config_detail.get("experiment_output_dir")
                report_file = run_detail.get("report_file")
                
                if report_file and os.path.exists(report_file):
                    try:
                        # 读取报告文件
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                        
                        # 提取关键指标
                        key_metrics = {}
                        
                        # 最新经济指标
                        if "economic_indicators" in report_data and report_data["economic_indicators"]:
                            indicators = report_data["economic_indicators"]
                            if isinstance(indicators, list) and len(indicators) > 0:
                                latest = indicators[-1]
                                
                                # 就业
                                if "employment_statistics" in latest:
                                    emp = latest["employment_statistics"]
                                    key_metrics["就业率"] = emp.get("labor_utilization_rate", 0)
                                    key_metrics["失业率"] = emp.get("household_unemployment_rate", 0)
                                
                                # 收入支出
                                if "income_expenditure_analysis" in latest:
                                    income_exp = latest["income_expenditure_analysis"]
                                    key_metrics["平均月收入"] = income_exp.get("average_monthly_income", 0)
                                    key_metrics["平均月支出"] = income_exp.get("average_monthly_expenditure", 0)
                                    key_metrics["储蓄率"] = income_exp.get("monthly_savings_rate", 0)
                                
                                # 财富分配
                                if "wealth_distribution" in latest:
                                    wealth = latest["wealth_distribution"]
                                    key_metrics["平均财富"] = wealth.get("average_wealth", 0)
                                    key_metrics["基尼系数"] = wealth.get("gini_coefficient", 0)
                        
                        analysis["详细结果"][config_name] = key_metrics
                        
                    except Exception as e:
                        analysis["详细结果"][config_name] = {"错误": f"读取报告失败: {str(e)}"}
                elif output_dir and os.path.exists(output_dir):
                    # 尝试从output目录查找最新的报告
                    try:
                        report_files = glob.glob(os.path.join(output_dir, "simulation_report_*.json"))
                        if report_files:
                            latest_report = max(report_files, key=os.path.getmtime)
                            with open(latest_report, 'r', encoding='utf-8') as f:
                                report_data = json.load(f)
                            
                            # 简化的指标提取
                            key_metrics = {}
                            if "economic_indicators" in report_data:
                                indicators = report_data["economic_indicators"]
                                if isinstance(indicators, list) and len(indicators) > 0:
                                    latest = indicators[-1]
                                    if "employment_statistics" in latest:
                                        key_metrics["就业率"] = latest["employment_statistics"].get("labor_utilization_rate", 0)
                                    if "wealth_distribution" in latest:
                                        key_metrics["基尼系数"] = latest["wealth_distribution"].get("gini_coefficient", 0)
                            
                            analysis["详细结果"][config_name] = key_metrics
                    except Exception as e:
                        analysis["详细结果"][config_name] = {"提示": f"无法读取output: {str(e)}"}
                else:
                    analysis["详细结果"][config_name] = {"提示": "实验成功但未找到报告文件"}
            elif status == "failed":
                # 提取失败信息
                run_log = run_detail.get("run_log", "")
                failure_info = {
                    "状态": "失败",
                    "持续时间": run_detail.get("duration_seconds", 0),
                }
                
                # 分析失败原因
                if "timeout" in run_log.lower() or run_detail.get("duration_seconds", 0) >= 3600:
                    failure_info["失败类型"] = "超时"
                    failure_info["建议"] = "减少num_households或num_iterations，或增加timeout时间"
                elif "error" in run_log.lower():
                    failure_info["失败类型"] = "运行错误"
                    # 提取错误信息的最后几行
                    log_lines = run_log.split('\n')
                    error_lines = [l for l in log_lines if 'error' in l.lower() or 'exception' in l.lower()]
                    if error_lines:
                        failure_info["错误摘要"] = error_lines[-1][:150]
                else:
                    failure_info["失败类型"] = "未知"
                    failure_info["日志摘要"] = run_log[-200:] if len(run_log) > 200 else run_log
                
                analysis["详细结果"][config_name] = failure_info
        
        # 汇总统计
        summary = {
            "配置组总数": total_configs,
            "完成数": completed_runs,
            "失败数": failed_runs,
            "成功率": f"{completed_runs/total_configs*100:.1f}%" if total_configs > 0 else "N/A",
            "是否包含详细分析": include_output_analysis and completed_runs > 0,
            "实验状态": "全部失败" if failed_runs == total_configs else ("部分成功" if completed_runs > 0 else "未运行"),
        }
        
        # 格式化输出 - 添加更友好的文本摘要
        response_text = f"<status>success</status>\n"
        response_text += f"<experiment_dir>{experiment_dir_abs}</experiment_dir>\n"
        
        # 添加可读的摘要
        response_text += "\n## 📊 实验概览\n\n"
        response_text += f"**实验名称**: {analysis['实验基本信息']['名称']}\n"
        response_text += f"**研究问题**: {analysis['实验基本信息']['研究问题']}\n"
        response_text += f"**假设**: {analysis['实验基本信息']['假设']}\n"
        response_text += f"**状态**: {summary['实验状态']} ({completed_runs}/{total_configs} 成功)\n\n"
        
        # 如果全部失败，提供诊断建议
        if failed_runs == total_configs:
            response_text += "## ⚠️ 实验状态：全部配置失败\n\n"
            response_text += "所有配置组均未成功完成，无法进行结果分析。\n\n"
            response_text += "### 失败详情：\n\n"
            for config_name, failure_info in analysis["详细结果"].items():
                if isinstance(failure_info, dict) and "失败类型" in failure_info:
                    response_text += f"**{config_name}**:\n"
                    response_text += f"- 失败类型: {failure_info.get('失败类型', '未知')}\n"
                    response_text += f"- 运行时长: {failure_info.get('持续时间', 0):.0f}秒\n"
                    if "建议" in failure_info:
                        response_text += f"- 建议: {failure_info['建议']}\n"
                    response_text += "\n"
            
            response_text += "### 💡 建议的解决方案：\n\n"
            response_text += "1. **减少规模**: 将 `num_households` 从 200 降至 50-100\n"
            response_text += "2. **缩短周期**: 将 `num_iterations` 从 24 降至 12\n"
            response_text += "3. **增加超时**: 将 timeout 从 3600秒 增至 7200秒\n"
            response_text += "4. **检查日志**: 查看具体卡在哪个环节（消费、招聘、创新等）\n\n"
        
        # 如果有成功的，展示对比
        elif completed_runs > 0:
            response_text += "## 📈 关键指标对比\n\n"
            # 这里会展示成功配置的指标对比
            pass
        
        # 添加JSON数据（供程序解析）
        response_text += f"\n<summary>{json.dumps(summary, ensure_ascii=False, indent=2)}</summary>\n"
        response_text += f"<analysis>{json.dumps(analysis, ensure_ascii=False, indent=2)}</analysis>\n"
        
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=response_text,
                ),
            ],
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>分析实验失败: {str(e)}</error>"
                        f"<traceback>{error_trace}</traceback>"
                    ),
                ),
            ],
        )


def read_experiment_manifest(
    experiment_dir: str,
) -> ToolResponse:
    """读取并分析实验项目的manifest.yaml文件。
    
    这个工具专门用于读取实验项目目录中的manifest.yaml，
    提取实验设计、配置、运行状态等信息，即使实验失败也能查看。
    
    Args:
        experiment_dir: 实验项目目录路径（包含manifest.yaml的目录）
    
    Returns:
        ToolResponse: 包含实验信息、配置、运行状态的详细分析
    """
    try:
        # 确保路径存在
        experiment_dir_abs = os.path.abspath(experiment_dir)
        if not os.path.exists(experiment_dir_abs):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>实验目录不存在: {experiment_dir_abs}</error>"
                        ),
                    ),
                ],
            )
        
        # 查找manifest文件
        manifest_path = os.path.join(experiment_dir_abs, "manifest.yaml")
        if not os.path.exists(manifest_path):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>未找到manifest.yaml文件: {manifest_path}</error>"
                            f"<hint>请确认这是一个实验项目目录</hint>"
                        ),
                    ),
                ],
            )
        
        # 读取manifest
        manifest = _load_manifest(manifest_path)
        if not manifest:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>manifest文件为空或格式错误</error>"
                        ),
                    ),
                ],
            )
        
        # 提取关键信息
        exp_info = manifest.get("experiment_info", {})
        metadata = manifest.get("metadata", {})
        configurations = manifest.get("configurations", {})
        runs = manifest.get("runs", {})
        
        # 构建分析结果
        analysis = {
            "实验基本信息": {
                "名称": exp_info.get("name", "未命名"),
                "描述": exp_info.get("description", "无"),
                "创建日期": exp_info.get("created_date", "未知"),
                "作者": exp_info.get("author", "未知"),
                "标签": exp_info.get("tags", []),
            },
            "研究设计": {
                "研究问题": metadata.get("research_question", "未定义"),
                "假设": metadata.get("hypothesis", "未定义"),
                "预期结果": metadata.get("expected_outcome", "未定义"),
            },
            "实验状态": {
                "当前状态": metadata.get("status", "unknown"),
                "开始时间": metadata.get("runtime", {}).get("start_time", "未开始"),
                "结束时间": metadata.get("runtime", {}).get("end_time", "未结束"),
                "总耗时(秒)": metadata.get("runtime", {}).get("duration_seconds", 0),
            },
            "配置组信息": {},
            "运行状态": {},
        }
        
        # 配置组详情
        for config_name, config_detail in configurations.items():
            analysis["配置组信息"][config_name] = {
                "参数变更": config_detail.get("parameters_changed", {}),
                "输出目录": config_detail.get("experiment_output_dir", "未定义"),
            }
        
        # 运行状态详情
        for run_name, run_detail in runs.items():
            status = run_detail.get("status", "unknown")
            analysis["运行状态"][run_name] = {
                "状态": status,
                "开始时间": run_detail.get("start_time", "未开始"),
                "结束时间": run_detail.get("end_time", "未结束"),
                "耗时(秒)": run_detail.get("duration_seconds", 0),
                "报告文件": run_detail.get("report_file", "未生成"),
                "日志文件": run_detail.get("log_file", "无"),
            }
            
            # 如果失败，添加失败原因
            if status == "failed":
                run_log = run_detail.get("run_log", "")
                if run_log:
                    # 截取关键错误信息
                    error_msg = run_log[:200] + "..." if len(run_log) > 200 else run_log
                    analysis["运行状态"][run_name]["失败原因"] = error_msg
        
        # 统计信息
        total_configs = len(configurations)
        total_runs = len(runs)
        completed_runs = sum(1 for r in runs.values() if r.get("status") == "completed")
        failed_runs = sum(1 for r in runs.values() if r.get("status") == "failed")
        
        summary = {
            "配置组总数": total_configs,
            "运行总数": total_runs,
            "完成数": completed_runs,
            "失败数": failed_runs,
            "成功率": f"{completed_runs/total_runs*100:.1f}%" if total_runs > 0 else "N/A",
        }
        
        # 格式化输出
        response_text = f"<status>success</status>\n"
        response_text += f"<experiment_dir>{experiment_dir_abs}</experiment_dir>\n"
        response_text += f"<manifest_path>{manifest_path}</manifest_path>\n"
        response_text += f"<summary>{json.dumps(summary, ensure_ascii=False, indent=2)}</summary>\n"
        response_text += f"<analysis>{json.dumps(analysis, ensure_ascii=False, indent=2)}</analysis>\n"
        
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=response_text,
                ),
            ],
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>读取manifest失败: {str(e)}</error>"
                        f"<traceback>{error_trace}</traceback>"
                    ),
                ),
            ],
        )


def analyze_experiment_directory(
    experiment_dir: str,
    experiment_name: str | None = None,
) -> ToolResponse:
    """Analyze an existing experiment directory using ExperimentAnalyzer.
    
    This tool loads and analyzes all metrics from an experiment output directory,
    including consumer metrics, firm metrics, innovation analysis, and macro metrics.
    
    Args:
        experiment_dir:
            Path to the experiment output directory (e.g., '/root/project/agentsociety-ecosim/output/encouraged')
        experiment_name:
            Optional name for the experiment. If not provided, will be derived from directory name.
    
    Returns:
        ToolResponse:
            A response containing comprehensive analysis results including:
            - Consumer metrics: total expenditure, total nutrition value, total satisfaction, total attribute value
            - Firm metrics: total revenue, total profit, productivity
            - Innovation metrics: innovation event counts, correlation with market share changes
            - Macro metrics: GDP, total revenue, total expenditure
    """
    try:
        from experiment_analyzer import ExperimentAnalyzer
        
        experiment_dir_abs = os.path.abspath(experiment_dir)
        if not os.path.exists(experiment_dir_abs):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Experiment directory not found: {experiment_dir_abs}</error>"
                        ),
                    ),
                ],
            )
        
        # Use directory name as experiment name if not provided
        if not experiment_name:
            experiment_name = os.path.basename(experiment_dir_abs.rstrip('/'))
        
        # Initialize analyzer
        analyzer = ExperimentAnalyzer(experiment_dir_abs)
        
        # Create analysis config
        from experiment_analyzer import AnalysisConfig
        config = AnalysisConfig(
            experiment_dir=experiment_dir_abs,
            experiment_name=experiment_name,
            metrics_to_analyze=['consumer_metrics', 'firm_metrics', 'innovation_analysis', 'macro_metrics']
        )
        
        # Perform comprehensive analysis
        metrics = analyzer.analyze_all_metrics(config)
        
        # Format results
        result_dict = {
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir_abs,
            "consumer_metrics": {
                "total_expenditure": metrics.consumer_metrics.total_expenditure if metrics.consumer_metrics else None,
                "total_nutrition_value": metrics.consumer_metrics.total_nutrition_value if metrics.consumer_metrics else None,
                "total_satisfaction_value": metrics.consumer_metrics.total_satisfaction_value if metrics.consumer_metrics else None,
                "total_attribute_value": metrics.consumer_metrics.total_attribute_value if metrics.consumer_metrics else None,
            } if metrics.consumer_metrics else None,
            "firm_metrics": {
                "total_profit": metrics.firm_metrics.total_profit if metrics.firm_metrics else None,
                "avg_profit": metrics.firm_metrics.avg_profit if metrics.firm_metrics else None,
                "total_revenue": metrics.macro_metrics.total_revenue if metrics.macro_metrics else None,
            } if metrics.firm_metrics or metrics.macro_metrics else None,
            "innovation_metrics": {
                "innovation_events_count": metrics.innovation_analysis.innovation_events_count if metrics.innovation_analysis else None,
                "market_share_correlation": metrics.innovation_analysis.innovation_market_share_correlation if metrics.innovation_analysis else None,
            } if metrics.innovation_analysis else None,
            "macro_metrics": {
                "gdp": metrics.macro_metrics.gdp if metrics.macro_metrics else None,
                "total_revenue": metrics.macro_metrics.total_revenue if metrics.macro_metrics else None,
                "total_expenditure": metrics.macro_metrics.total_expenditure if metrics.macro_metrics else None,
            } if metrics.macro_metrics else None,
        }
        
        response_text = f"<status>success</status>\n"
        response_text += f"<experiment_name>{experiment_name}</experiment_name>\n"
        response_text += f"<experiment_dir>{experiment_dir_abs}</experiment_dir>\n"
        response_text += f"<analysis_results>{json.dumps(result_dict, indent=2, ensure_ascii=False)}</analysis_results>\n"
        
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=response_text,
                ),
            ],
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to analyze experiment directory '{experiment_dir}': {str(e)}</error>"
                        f"<traceback>{error_trace}</traceback>"
                    ),
                ),
            ],
        )


def compare_experiments(
    experiment1_name: str,
    experiment2_name: str,
    metrics_to_compare: list[str] | None = None,
    manifest_path: str | None = None,
) -> ToolResponse:
    """Compare key metrics between two simulation experiments.
    
    This tool reads reports from two experiments and provides a detailed
    comparison of their key economic indicators.
    
    Args:
        experiment1_name:
            Name of the first experiment (control group).
        experiment2_name:
            Name of the second experiment (treatment group).
        metrics_to_compare:
            Optional list of specific metrics to compare. If not provided,
            compares all available key metrics.
    
    Returns:
        ToolResponse:
            A response containing side-by-side comparison and difference analysis.
    """
    # Read both reports
    report1_resp = read_simulation_report(experiment_name=experiment1_name)
    report2_resp = read_simulation_report(experiment_name=experiment2_name)
    
    if "error" in report1_resp.content[0].text:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to read experiment 1 ({experiment1_name}): "
                        f"{report1_resp.content[0].text}</error>"
                    ),
                ),
            ],
        )
    
    if "error" in report2_resp.content[0].text:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to read experiment 2 ({experiment2_name}): "
                        f"{report2_resp.content[0].text}</error>"
                    ),
                ),
            ],
        )
    
    # Extract metrics from responses
    try:
        metrics1_text = re.search(r"<key_metrics>(.*?)</key_metrics>", report1_resp.content[0].text, re.DOTALL)
        metrics2_text = re.search(r"<key_metrics>(.*?)</key_metrics>", report2_resp.content[0].text, re.DOTALL)
        
        if not metrics1_text or not metrics2_text:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"<status>error</status>"
                            f"<error>Failed to extract metrics from reports</error>"
                        ),
                    ),
                ],
            )
        
        metrics1 = json.loads(metrics1_text.group(1))
        metrics2 = json.loads(metrics2_text.group(1))
        
        # Perform comparison
        comparison = {
            "experiment1": experiment1_name,
            "experiment2": experiment2_name,
            "comparisons": {},
        }
        
        def compare_values(path: str, val1: Any, val2: Any):
            """Recursively compare nested dictionaries."""
            if isinstance(val1, dict) and isinstance(val2, dict):
                for key in set(val1.keys()) | set(val2.keys()):
                    if key in val1 and key in val2:
                        compare_values(f"{path}.{key}", val1[key], val2[key])
                    elif key in val1:
                        comparison["comparisons"][f"{path}.{key}"] = {
                            "experiment1": val1[key],
                            "experiment2": "N/A",
                            "difference": "N/A",
                            "percent_change": "N/A",
                        }
                    else:
                        comparison["comparisons"][f"{path}.{key}"] = {
                            "experiment1": "N/A",
                            "experiment2": val2[key],
                            "difference": "N/A",
                            "percent_change": "N/A",
                        }
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                comparison["comparisons"][path] = {
                    "experiment1": val1,
                    "experiment2": val2,
                    "difference": diff,
                    "percent_change": pct_change,
                }
            else:
                comparison["comparisons"][path] = {
                    "experiment1": val1,
                    "experiment2": val2,
                    "difference": "N/A",
                    "percent_change": "N/A",
                }
        
        # Compare all metrics
        for key in set(metrics1.keys()) | set(metrics2.keys()):
            if key in metrics1 and key in metrics2:
                compare_values(key, metrics1[key], metrics2[key])
            elif key in metrics1:
                comparison["comparisons"][key] = {
                    "experiment1": metrics1[key],
                    "experiment2": "N/A",
                }
            else:
                comparison["comparisons"][key] = {
                    "experiment1": "N/A",
                    "experiment2": metrics2[key],
                }
        
        response_parts = [
            f"<status>success</status>",
            f"<experiment1>{experiment1_name}</experiment1>",
            f"<experiment2>{experiment2_name}</experiment2>",
            f"<comparison>{json.dumps(comparison, indent=2, ensure_ascii=False)}</comparison>",
        ]

        if manifest_path:
            _record_manifest_comparison(manifest_path, comparison)
            response_parts.append(f"<manifest_updated>{manifest_path}</manifest_updated>")
        
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="\n".join(response_parts),
                ),
            ],
        )
        
    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<status>error</status>"
                        f"<error>Failed to compare experiments: {e}</error>"
                    ),
                ),
            ],
        )


def build_design_agent(api_key: str) -> tuple[ReActAgent, UserAgent]:
    """Instantiate the Design Agent and corresponding user agent."""
    toolkit = Toolkit()
    tool_call_logger = ToolCallLogger()

    def register_tool(fn: Callable):
        toolkit.register_tool_function(_wrap_tool_function(fn, tool_call_logger))

    # Parameter query tools (should be called BEFORE designing experiments)
    register_tool(get_available_parameters)
    register_tool(get_parameter_info)
    # Experiment management tools
    register_tool(init_experiment_manifest)
    register_tool(create_yaml_from_template)
    register_tool(run_simulation)
    register_tool(read_simulation_report)
    register_tool(compare_experiments)
    # Experiment analysis tools
    register_tool(analyze_experiment_from_manifest)  # 新增: 完整分析（manifest + output）
    register_tool(read_experiment_manifest)  # 读取manifest
    register_tool(analyze_experiment_directory)  # 分析output目录
    
    # Knowledge base tools (academic paper retrieval)
    try:
        import sys
        # Use relative path: database is now in the same parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))  # /root/project/agentsociety-ecosim/economist
        database_path = os.path.abspath(os.path.join(current_dir, "../database"))
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
        ) -> ToolResponse:
            """
            查询学术论文知识库，获取与研究问题相关的学术文献。
            
            **CRITICAL: Query MUST be in English!** The SPECTER2 embedding model is optimized for English text.
            If the user provides a Chinese query, you MUST translate it to English before calling this function.
            
            Args:
                query: Natural language query describing the research topic (**MUST be in English**)
                top_k: Number of results to return (1-20)
                journals: Journal filter (optional)
                year_start: Start year (optional, 0 means no filter)
                year_end: End year (optional, 0 means no filter)
                doc_type: Document type (currently **not used for filtering** to ensure best results)
            
            Returns:
                Response containing relevant paper information
            """
            result = _query_kb(
                query=query,
                top_k=top_k,
                journals=journals if journals else None,
                year_start=year_start if year_start > 0 else None,
                year_end=year_end if year_end > 0 else None,
                # 不按照 doc_type 进行限制，始终在所有论文块上检索
                doc_type=None,
            )
            
            if result.get("status") == "success":
                results_json = json.dumps(result.get("results", []), ensure_ascii=False, indent=2)
                response_text = (
                    f"<status>success</status>"
                    f"<query>{query}</query>"
                    f"<total_found>{result.get('total_found', 0)}</total_found>"
                    f"<results>{results_json}</results>"
                )
            else:
                response_text = (
                    f"<status>error</status>"
                    f"<error>{result.get('error', 'Unknown error')}</error>"
                )
            
            return ToolResponse(content=[TextBlock(type="text", text=response_text)])
        
        def find_similar_papers(
            title: str,
            abstract: str = "",
            top_k: int = 5,
        ) -> ToolResponse:
            """
            查找与给定论文相似的学术论文。
            
            Args:
                title: 论文标题
                abstract: 论文摘要（可选）
                top_k: 返回结果数量
            
            Returns:
                相似论文列表
            """
            result = _find_similar(title=title, abstract=abstract, top_k=top_k)
            
            if result.get("status") == "success":
                results_json = json.dumps(result.get("similar_papers", []), ensure_ascii=False, indent=2)
                response_text = (
                    f"<status>success</status>"
                    f"<query_title>{title}</query_title>"
                    f"<total_found>{result.get('total_found', 0)}</total_found>"
                    f"<similar_papers>{results_json}</similar_papers>"
                )
            else:
                response_text = (
                    f"<status>error</status>"
                    f"<error>{result.get('error', 'Unknown error')}</error>"
                )
            
            return ToolResponse(content=[TextBlock(type="text", text=response_text)])
        
        def get_paper_details(
            paper_id: str,
            include_similar: bool = True,
            similar_count: int = 3,
        ) -> ToolResponse:
            """
            获取整篇论文的上下文（摘要 + 章节 + 相似论文）。
            """
            result = _get_paper_details(paper_id=paper_id)
            if result.get("status") == "success":
                response_text = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                response_text = (
                    f"<status>error</status>"
                    f"<error>{result.get('error', 'Unknown error')}</error>"
                )
            return ToolResponse(content=[TextBlock(type="text", text=response_text)])
        
        register_tool(query_knowledge_base)
        register_tool(find_similar_papers)
        register_tool(get_paper_details)
    except ImportError as e:
        print(f"Warning: Knowledge base tools not available: {e}")

    agent = ReActAgent(
        name="DesignAgent",
        sys_prompt=_build_sys_prompt(),
        model=OpenAIChatModel(
            model_name="gpt-5", # deepseek-ai/DeepSeek-V3.2
            api_key=api_key,
            client_args={
                "base_url": "http://35.220.164.252:3888/v1/",
            },
            stream=False,  # 关闭流式响应以避免 tool_calls 消息序列不完整的问题
            # 如果遇到 "tool_calls must be followed by tool messages" 错误，
            # 通常是因为流式响应中断导致消息序列不完整
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
    )
    agent.register_instance_hook(
        "pre_print",
        "normalize_finish_response",
        _normalize_finish_response,
    )
    agent.tool_call_logger = tool_call_logger

    user = UserAgent(name="user")
    return agent, user


async def main():
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY (or DASHSCOPE_API_KEY) environment variable."
        )
    agent, user = build_design_agent(api_key)

    # Send initial welcome message
    welcome_msg = Msg(
        name="user",
        role="user",
        content="Hello, could you introduce what research you can help me with?"
    )
    msg = await agent(welcome_msg)
    print(msg.get_text_content())
    
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)

if __name__ == "__main__":
    asyncio.run(main())
