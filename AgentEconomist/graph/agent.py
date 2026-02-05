"""
LangGraph Agent 构建模块

构建完整的 Agent Economist 工作流。
"""

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import create_init_node, create_call_model_node, create_update_fs_state_node
from ..prompts.system_prompt import build_system_prompt
from ..config import Config
from .. import tools


def build_economist_graph():
    """构建 Agent Economist 的 LangGraph 工作流"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.BASE_URL,
        temperature=Config.LLM_TEMPERATURE,
    )
    
    # Create tools
    tool_list = [
        StructuredTool.from_function(
            func=tools.get_available_parameters,
            name="get_available_parameters",
            description="Get all available simulation parameters. CALL THIS BEFORE designing experiments!"
        ),
        StructuredTool.from_function(
            func=tools.get_parameter_info,
            name="get_parameter_info",
            description="Get detailed information about a specific parameter."
        ),
        StructuredTool.from_function(
            func=tools.init_manifest,
            name="init_manifest",
            description="Initialize experiment directory and manifest. CALL THIS FIRST when user asks a research question!"
        ),
        StructuredTool.from_function(
            func=tools.update_experiment_metadata,
            name="update_experiment_metadata",
            description="Update experiment metadata (name, description, hypothesis, etc.). Can be called multiple times."
        ),
        StructuredTool.from_function(
            func=tools.create_yaml_from_template,
            name="create_yaml_from_template",
            description="Create a YAML configuration file from template with modified parameters."
        ),
        StructuredTool.from_function(
            func=tools.modify_yaml_parameters,
            name="modify_yaml_parameters",
            description=(
                "Modify parameters for multiple configuration groups at once. "
                "Use this when you need to change parameters in existing configs. "
                "Pass group_names as comma-separated string (e.g. 'control,treatment'). "
                "Example: modify_yaml_parameters(manifest_path='...', group_names='control,treatment', "
                "parameter_changes='{\"system_scale.num_iterations\": 2}')"
            )
        ),
        StructuredTool.from_function(
            func=tools.run_simulation,
            name="run_simulation",
            description=(
                "Run economic simulation(s). "
                "IMPORTANT: Use run_all=True to run all configs sequentially (RECOMMENDED to avoid resource conflicts). "
                "Example: run_simulation(manifest_path='/path/to/manifest.yaml', run_all=True)"
            )
        ),
        StructuredTool.from_function(
            func=tools.read_simulation_report,
            name="read_simulation_report",
            description=(
                "Read simulation_report.json for a quick summary. "
                "For detailed analysis, use analyze_experiment_directory instead."
            )
        ),
        StructuredTool.from_function(
            func=tools.compare_experiments,
            name="compare_experiments",
            description=(
                "Compare economic metrics across experiment groups. "
                "Extracts data from output_dir/data/ (economic_metrics_history.json, "
                "household_monthly_metrics.json, firm_monthly_metrics.json, etc.) "
                "and calculates percent changes for employment, inequality, income, firm performance."
            )
        ),
        StructuredTool.from_function(
            func=tools.analyze_experiment_directory,
            name="analyze_experiment_directory",
            description=(
                "Analyze a single experiment's results in detail. "
                "Extracts key metrics from output_dir/data/ including employment stats, "
                "income/expenditure, inequality (Gini), consumption structure, and trends."
            )
        ),
        StructuredTool.from_function(
            func=tools.query_knowledge_base,
            name="query_knowledge_base",
            description=(
                "Query academic literature database. "
                "CRITICAL: query MUST be in English! "
                "Results are automatically saved to manifest if manifest_path is provided."
            )
        ),
    ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tool_list)
    
    # System prompt
    system_prompt = build_system_prompt()
    
    # Create nodes
    init_node = create_init_node(system_prompt)
    call_model = create_call_model_node(llm_with_tools, system_prompt)
    update_fs_state_node = create_update_fs_state_node()
    
    # Build Graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("init_node", init_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", ToolNode(tool_list))
    workflow.add_node("update_fs_state", update_fs_state_node)
    
    # Set entry point
    workflow.set_entry_point("init_node")
    
    # Define edges
    workflow.add_edge("init_node", "call_model")
    workflow.add_edge("tools", "update_fs_state")
    workflow.add_edge("update_fs_state", "call_model")
    
    # Conditional edge from call_model
    workflow.add_conditional_edges(
        "call_model",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    
    # Compile graph with checkpointer
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export graph for langgraph.json
graph = build_economist_graph()
