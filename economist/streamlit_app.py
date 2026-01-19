"""
Design Agent Streamlit Web Interface
美观的 Web 界面用于与 Design Agent 交互
基于项目的多 Agent 管理系统
"""

import streamlit as st
import os
import yaml
import json
import glob
import re
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List
from agentscope.message import Msg

# Page configuration
st.set_page_config(
    page_title="Design Agent - Economic System Design Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .experiment-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-planned { background-color: #e3f2fd; color: #1976d2; }
    .status-running { background-color: #fff3e0; color: #f57c00; }
    .status-completed { background-color: #e8f5e9; color: #388e3c; }
    .status-failed { background-color: #ffebee; color: #d32f2f; }
    .status-analysis_pending { background-color: #fff3e0; color: #f57c00; }
    .agent-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .agent-item:hover {
        border-color: #1f77b4;
        background-color: #f0f7ff;
    }
    .agent-item.selected {
        border-color: #1f77b4;
        background-color: #e3f2fd;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if "agents" not in st.session_state:
    st.session_state.agents: Dict[str, any] = {}  # {project_dir: agent_instance}
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages: Dict[str, List[dict]] = {}  # {project_dir: [messages]}
if "current_agent_dir" not in st.session_state:
    st.session_state.current_agent_dir: Optional[str] = None
if "project_list" not in st.session_state:
    st.session_state.project_list: List[dict] = []
if "agent_stage" not in st.session_state:
    st.session_state.agent_stage: Dict[str, Optional[str]] = {}

# 项目根目录（使用动态路径）
PROJECT_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHAT_HISTORY_FILENAME = "chat_history.json"


def get_chat_history_path(project_dir: str) -> Path:
    return Path(project_dir) / CHAT_HISTORY_FILENAME


def load_chat_history(project_dir: str) -> List[dict]:
    history_path = get_chat_history_path(project_dir)
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f) or []
            if isinstance(data, list):
                return [
                    msg for msg in data
                    if isinstance(msg, dict) and msg.get("role") and msg.get("content")
                ]
        except Exception as e:
            st.warning(f"Failed to load chat history for {project_dir}: {e}")
    return []


def save_chat_history(project_dir: str, messages: List[dict]) -> None:
    history_path = get_chat_history_path(project_dir)
    try:
        os.makedirs(history_path.parent, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Failed to save chat history for {project_dir}: {e}")


def update_stage_from_history(project_dir: str) -> None:
    messages = st.session_state.agent_messages.get(project_dir, [])
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("stage"):
            st.session_state.agent_stage[project_dir] = message["stage"]
            return
    st.session_state.agent_stage.pop(project_dir, None)


def infer_workflow_stage(content: str) -> Optional[str]:
    checks = [
        ("awaiting_hypothesis_confirmation", [
            "does this hypothesis",
            "should i proceed with designing the experiment",
        ]),
        ("awaiting_config_creation", [
            "should i proceed to create the configuration files",
        ]),
        ("awaiting_run_approval", [
            "configs are ready. should i start running the simulations",
        ]),
        ("awaiting_analysis_confirmation", [
            "should i proceed with the analysis",
        ]),
        ("awaiting_verification", [
            "would you like me to proceed with hypothesis verification",
        ]),
    ]
    lowered = content.lower()
    for stage, phrases in checks:
        if any(phrase in lowered for phrase in phrases):
            return stage
    return None


def tool_event_to_message(event: dict) -> dict:
    return {
        "role": "tool",
        "tool_name": event.get("tool_name", "unknown_tool"),
        "status": event.get("status", "success"),
        "input": event.get("input"),
        "output": event.get("output"),
        "error": event.get("error"),
        "timestamp": event.get("timestamp"),
    }


def parse_kb_results(tool_events: List[dict]) -> List[dict]:
    """从工具事件中提取知识库检索结果（query_knowledge_base）。"""
    results: List[dict] = []
    for evt in tool_events or []:
        if evt.get("tool_name") != "query_knowledge_base":
            continue
        if evt.get("status") != "success":
            continue
        output = (
            evt.get("output")
            or evt.get("content")
            or evt.get("text")
            or ""
        )
        # 解析 <results>...</results> 内的 JSON
        start = output.find("<results>")
        end = output.find("</results>")
        if start != -1 and end != -1 and end > start:
            payload = output[start + len("<results>"):end].strip()
            try:
                data = json.loads(payload)
                if isinstance(data, list):
                    results.extend(data)
            except Exception:
                continue
    return results


def render_kb_results(results: List[dict]) -> None:
    """以可折叠卡片方式展示检索结果"""
    if not results:
        return
    with st.expander(f"📚 Retrieved Papers / Sections ({len(results)})", expanded=True):
        for idx, item in enumerate(results, 1):
            title = item.get("title", "Untitled")
            journal = item.get("journal", "")
            publish_time = item.get("publish_time", "")
            score = item.get("score", "")
            doc_type = item.get("doc_type", "")
            section_title = item.get("section_title") or ""
            preview = item.get("text") or ""
            paper_id = item.get("paper_id") or ""
            pdf_link = item.get("pdf_link") or ""
            
            st.markdown(f"**{idx}. {title}**")
            meta_parts = []
            if journal:
                meta_parts.append(journal)
            if publish_time:
                meta_parts.append(publish_time)
            if score != "":
                meta_parts.append(f"score {score}")
            if doc_type:
                meta_parts.append(doc_type)
            if section_title:
                meta_parts.append(f"section: {section_title}")
            if meta_parts:
                st.caption(" · ".join(meta_parts))
            if preview:
                st.write(preview[:300] + ("..." if len(preview) > 300 else ""))
            link_parts = []
            if paper_id:
                link_parts.append(f"`paper_id`: {paper_id}")
                st.caption("如果该段落有价值，可用 get_paper_details(paper_id) 拉取全文摘要+章节。")
            if pdf_link:
                link_parts.append(f"[PDF]({pdf_link})")
            if link_parts:
                st.caption(" | ".join(link_parts))
            st.markdown("---")


def generate_project_name_from_message(message: str) -> str:
    """从用户消息中提取关键词，生成英文项目名"""
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', ' ', message.lower())
    
    # 常见经济研究关键词映射
    keyword_map = {
        '政策': 'policy',
        '创新': 'innovation',
        '政府': 'government',
        '税收': 'tax',
        '就业': 'employment',
        '失业': 'unemployment',
        '收入': 'income',
        '财富': 'wealth',
        '分配': 'distribution',
        '经济': 'economic',
        '研究': 'study',
        '实验': 'experiment',
        '影响': 'impact',
        '作用': 'effect',
        '分析': 'analysis',
    }
    
    # 提取关键词
    words = text.split()
    keywords = []
    for word in words:
        if word in keyword_map:
            keywords.append(keyword_map[word])
        elif len(word) > 3 and word.isalpha():
            keywords.append(word)
    
    # 如果提取到关键词，组合成项目名
    if keywords:
        # 取前3个关键词，用下划线连接
        project_name = '_'.join(keywords[:3])
        # 确保名称符合文件系统规范
        project_name = re.sub(r'[^a-z0-9_]', '_', project_name)
        project_name = re.sub(r'_+', '_', project_name).strip('_')
        if len(project_name) > 50:
            project_name = project_name[:50]
        return project_name if project_name else "research_study"
    
    # 如果没有提取到关键词，使用默认名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"research_study_{timestamp}"


def create_project_auto(project_name: str, research_question: str = "") -> Optional[str]:
    """自动创建项目（在 experiments 目录下）"""
    try:
        # 确保 experiments 目录存在
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 生成项目目录路径
        project_dir_name = project_name
        project_path = EXPERIMENTS_DIR / project_dir_name
        
        # 如果目录已存在，添加序号
        counter = 1
        while project_path.exists():
            project_dir_name = f"{project_name}_{counter}"
            project_path = EXPERIMENTS_DIR / project_dir_name
            counter += 1
        
        # 创建项目目录
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 manifest.yaml
        from design_agent import init_experiment_manifest
        init_experiment_manifest(
            experiment_dir=str(project_path),
            experiment_name=project_name,  # 修复：使用 experiment_name 而不是 project_name
            research_question=research_question,
            hypothesis="",
        )
        
        return str(project_path)
    except Exception as e:
        st.error(f"自动创建项目失败: {e}")
        return None


def rename_project(old_dir: str, new_name: str) -> Optional[str]:
    """重命名项目（包括目录名和 manifest 中的名称）"""
    try:
        old_path = Path(old_dir)
        if not old_path.exists():
            return None
        
        # 确保新名称符合文件系统规范
        new_dir_name = re.sub(r'[^a-z0-9_]', '_', new_name.lower())
        new_dir_name = re.sub(r'_+', '_', new_dir_name).strip('_')
        if not new_dir_name:
            new_dir_name = "project"
        
        # 确定新目录路径（保持在 experiments 目录下）
        if old_path.parent == EXPERIMENTS_DIR:
            # 如果原目录在 experiments 下，新目录也在 experiments 下
            new_path = EXPERIMENTS_DIR / new_dir_name
        else:
            # 如果原目录不在 experiments 下，移动到 experiments 下
            new_path = EXPERIMENTS_DIR / new_dir_name
        
        # 如果新目录已存在，添加序号
        counter = 1
        while new_path.exists() and new_path != old_path:
            new_dir_name = f"{new_dir_name}_{counter}"
            new_path = EXPERIMENTS_DIR / new_dir_name
            counter += 1
        
        # 重命名目录
        if old_path != new_path:
            shutil.move(str(old_path), str(new_path))
        
        # 更新 manifest.yaml 中的名称
        manifest_path = new_path / "manifest.yaml"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f)
            
            if manifest:
                manifest["experiment_info"]["name"] = new_name
                manifest["experiment_info"]["directory"] = str(new_path)
                
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(manifest, f, allow_unicode=True, default_flow_style=False)
        
        return str(new_path)
    except Exception as e:
        st.error(f"重命名项目失败: {e}")
        return None


def load_project_directories() -> List[dict]:
    """加载所有包含 manifest.yaml 的项目目录（优先从 experiments 目录）"""
    projects = []
    
    # 优先查找 experiments 目录下的项目
    experiments_projects = []
    if EXPERIMENTS_DIR.exists():
        for manifest_file in EXPERIMENTS_DIR.rglob("manifest.yaml"):
            project_dir = manifest_file.parent
            
            # 只统计直接包含 manifest.yaml 的目录（避免嵌套目录）
            has_child_manifest = any(
                (p / "manifest.yaml").exists() 
                for p in project_dir.iterdir() 
                if p.is_dir()
            )
            
            if not has_child_manifest:
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = yaml.safe_load(f)
                    
                    if manifest:
                        exp_info = manifest.get("experiment_info", {})
                        metadata = manifest.get("metadata", {})
                        
                        project_name = project_dir.name
                        relative_path = project_dir.relative_to(PROJECT_ROOT)
                        display_name = str(relative_path).replace("/", " / ")
                        
                        experiments_projects.append({
                            "dir": str(project_dir),
                            "name": exp_info.get("name", project_name),
                            "display_name": display_name,
                            "status": metadata.get("status", "unknown"),
                            "created_date": exp_info.get("created_date", ""),
                            "research_question": metadata.get("research_question", ""),
                            "manifest_path": str(manifest_file),
                            "manifest": manifest
                        })
                except Exception as e:
                    st.error(f"加载项目目录失败 {manifest_file}: {e}")
    
    # 也查找其他目录下的项目（兼容旧项目）
    other_projects = []
    for manifest_file in PROJECT_ROOT.rglob("manifest.yaml"):
        project_dir = manifest_file.parent
        
        # 跳过 experiments 目录（已经处理过）
        if EXPERIMENTS_DIR in project_dir.parents or project_dir == EXPERIMENTS_DIR:
            continue
        
        has_child_manifest = any(
            (p / "manifest.yaml").exists() 
            for p in project_dir.iterdir() 
            if p.is_dir()
        )
        
        if not has_child_manifest:
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = yaml.safe_load(f)
                
                if manifest:
                    exp_info = manifest.get("experiment_info", {})
                    metadata = manifest.get("metadata", {})
                    
                    project_name = project_dir.name
                    relative_path = project_dir.relative_to(PROJECT_ROOT)
                    display_name = str(relative_path).replace("/", " / ")
                    
                    other_projects.append({
                        "dir": str(project_dir),
                        "name": exp_info.get("name", project_name),
                        "display_name": display_name,
                        "status": metadata.get("status", "unknown"),
                        "created_date": exp_info.get("created_date", ""),
                        "research_question": metadata.get("research_question", ""),
                        "manifest_path": str(manifest_file),
                        "manifest": manifest
                    })
            except Exception as e:
                st.error(f"加载项目目录失败 {manifest_file}: {e}")
    
    # 合并项目列表，experiments 目录下的项目优先
    projects = experiments_projects + other_projects
    
    return sorted(projects, key=lambda x: x["created_date"] or "", reverse=True)


def load_experiments_in_project(project_dir: str) -> List[dict]:
    """加载指定项目目录下的所有实验（manifest.yaml）"""
    project_path = Path(project_dir)
    experiments = []
    
    # 查找该目录及其子目录中的所有 manifest.yaml
    for manifest_file in project_path.rglob("manifest.yaml"):
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f)
            
            if manifest:
                exp_info = manifest.get("experiment_info", {})
                metadata = manifest.get("metadata", {})
                
                # 计算相对于项目目录的相对路径
                rel_path = manifest_file.parent.relative_to(project_path)
                
                experiments.append({
                    "path": str(manifest_file.parent),
                    "rel_path": str(rel_path) if rel_path != Path(".") else ".",
                    "name": exp_info.get("name", manifest_file.parent.name),
                    "status": metadata.get("status", "unknown"),
                    "created_date": exp_info.get("created_date", ""),
                    "research_question": metadata.get("research_question", ""),
                    "manifest": manifest
                })
        except Exception as e:
            st.error(f"加载实验失败 {manifest_file}: {e}")
    
    return sorted(experiments, key=lambda x: x["created_date"] or "", reverse=True)


def render_status_badge(status: str):
    """渲染状态标签"""
    status_map = {
        "planned": ("planning", "status-planned"),
        "running": ("running", "status-running"),
        "completed": ("completed", "status-completed"),
        "failed": ("failed", "status-failed"),
        "analysis_pending": ("analysis_pending", "status-running"),
    }
    label, css_class = status_map.get(status, ("unknown", "status-planned"))
    return f'<span class="status-badge {css_class}">{label}</span>'


def display_manifest(manifest: dict):
    """Display Manifest content"""
    st.subheader("📋 Experiment Manifest")
    
    # Experiment information
    exp_info = manifest.get("experiment_info", {})
    metadata = manifest.get("metadata", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Experiment Name", exp_info.get("name", "N/A"))
    with col2:
        st.metric("Created Date", exp_info.get("created_date", "N/A"))
    with col3:
        status = metadata.get("status", "unknown")
        st.markdown(f"**Status:** {render_status_badge(status)}", unsafe_allow_html=True)
    
    # Research information
    st.markdown("### Research Information")
    st.text_area("Research Question", metadata.get("research_question", ""), height=80, disabled=True, key="research_question")
    st.text_area("Hypothesis", metadata.get("hypothesis", ""), height=80, disabled=True, key="hypothesis")
    st.text_area("Expected Outcome", metadata.get("expected_outcome", ""), height=80, disabled=True, key="expected_outcome")
    
    # Configuration information
    configurations = manifest.get("configurations", {})
    if configurations:
        st.markdown("### Experiment Configurations")
        for label, config in configurations.items():
            with st.expander(f"📁 {label}"):
                st.code(f"Path: {config.get('path', 'N/A')}")
                params = config.get("parameters_changed", {})
                if params:
                    st.json(params)
    
    # Run status
    runs = manifest.get("runs", {})
    if runs:
        st.markdown("### Run Status")
        for label, run in runs.items():
            with st.expander(f"🔬 {label}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", run.get("status", "N/A"))
                    if run.get("start_time"):
                        st.caption(f"Start: {run['start_time']}")
                    if run.get("end_time"):
                        st.caption(f"End: {run['end_time']}")
                with col2:
                    if run.get("duration_seconds"):
                        st.metric("Duration", f"{run['duration_seconds']:.1f} seconds")
                    if run.get("log_file"):
                        if os.path.exists(run["log_file"]):
                            with open(run["log_file"], 'r', encoding='utf-8') as f:
                                log_content = f.read()
                            st.download_button(
                                "Download Log",
                                log_content,
                                file_name=os.path.basename(run["log_file"]),
                                mime="text/plain"
                            )
                
                # Key metrics
                metrics = run.get("key_metrics", {})
                if metrics:
                    st.markdown("**Key Metrics:**")
                    st.json(metrics)
    
    # Results summary
    results = manifest.get("results_summary", {})
    if results and results.get("conclusion"):
        st.markdown("### Experiment Results")
        st.text_area("Conclusion", results.get("conclusion", ""), height=150, disabled=True, key="conclusion")
        if results.get("comparison"):
            st.markdown("**Comparison Results:**")
            st.json(results.get("comparison"))


def visualize_metrics(manifest: dict):
    """Visualize key metrics"""
    runs = manifest.get("runs", {})
    if not runs:
        st.info("No run data available")
        return
    
    # Collect metrics from all runs
    metrics_data = []
    for label, run in runs.items():
        metrics = run.get("key_metrics", {})
        if metrics:
            summary = metrics.get("summary", {})
            if summary:
                row = {"run": label}
                row.update(summary)
                metrics_data.append(row)
    
    if not metrics_data:
        st.info("No metrics data available")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Select metrics to visualize
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        st.info("No numeric metrics available for visualization")
        return
    
    st.subheader("📊 Metrics Comparison")
    
    # Select metrics
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    )
    
    if selected_metrics:
        # Bar chart
        fig = go.Figure()
        for metric in selected_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df["run"],
                y=df[metric]
            ))
        
        fig.update_layout(
            title="Metrics Comparison",
            xaxis_title="Experiment Group",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(df[["run"] + selected_metrics], use_container_width=True)


def get_or_create_agent(project_dir: str):
    """Get or create Agent instance for specified project"""
    if project_dir not in st.session_state.agents:
        try:
            from streamlit_agent_wrapper import create_agent_for_streamlit
            agent_wrapper = create_agent_for_streamlit()
            st.session_state.agents[project_dir] = agent_wrapper
            
            # 初始化消息列表（如果不存在）
            if project_dir not in st.session_state.agent_messages:
                st.session_state.agent_messages[project_dir] = load_chat_history(project_dir)
            update_stage_from_history(project_dir)
            
            # 如果没有历史记录，发送欢迎消息
            if not st.session_state.agent_messages[project_dir]:
                try:
                    welcome_response = agent_wrapper.process_sync("你好，请介绍一下你可以帮助我做什么研究？")
                    if welcome_response:
                        st.session_state.agent_messages[project_dir].append({
                            "role": "user",
                            "content": "你好，请介绍一下你可以帮助我做什么研究？"
                        })
                        st.session_state.agent_messages[project_dir].append({
                            "role": "assistant",
                            "content": welcome_response,
                            "stage": infer_workflow_stage(welcome_response),
                        })
                        st.session_state.agent_stage[project_dir] = infer_workflow_stage(welcome_response)
                        save_chat_history(project_dir, st.session_state.agent_messages[project_dir])
                except Exception as e:
                    st.warning(f"发送欢迎消息时出错: {str(e)}")
        except Exception as e:
            st.error(f"初始化 Agent 失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    return st.session_state.agents[project_dir]


def main():
    """Main function"""
    # Title
    st.markdown('<div class="main-header">🧪 Agent Economist </div>', unsafe_allow_html=True)
    
    # Sidebar - Agent project list
    with st.sidebar:
        st.header("🤖 Agent Projects")
        
        # Refresh button
        if st.button("🔄 Refresh Project List", use_container_width=True):
            st.session_state.project_list = load_project_directories()
            st.rerun()
        
        # New conversation button
        if st.button("🆕 New Conversation", use_container_width=True):
            previous_dir = st.session_state.get("current_agent_dir")
            if previous_dir:
                st.session_state.agent_stage.pop(previous_dir, None)
            st.session_state.current_agent_dir = None
            st.session_state.show_rename = False
            st.session_state.pop("selected_experiment", None)
            st.rerun()
        
        # Load project list
        if not st.session_state.project_list:
            st.session_state.project_list = load_project_directories()
        
        st.divider()
        
        # Project list
        st.subheader("Project List")
        if st.session_state.project_list:
            for project in st.session_state.project_list:
                project_dir = project["dir"]
                is_selected = st.session_state.current_agent_dir == project_dir
                
                # Display project information
                with st.container():
                    if is_selected:
                        st.markdown(f"**✓ {project['name']}**")
                    else:
                        st.markdown(f"**{project['name']}**")
                    
                    # Display status
                    status = project.get("status", "unknown")
                    st.markdown(render_status_badge(status), unsafe_allow_html=True)
                    
                    # Display research question (if available)
                    if project.get("research_question"):
                        st.caption(project["research_question"][:50] + "..." if len(project["research_question"]) > 50 else project["research_question"])
                    
                    # Select button
                    if st.button(f"Select", key=f"select_{project_dir}", use_container_width=True):
                        st.session_state.current_agent_dir = project_dir
                        st.rerun()
                    
                    st.divider()
        else:
            st.info("No projects available. Start chatting to automatically create a project.")
    
    # 主内容区
    if st.session_state.current_agent_dir:
        # 显示当前选中的项目信息
        current_project = next(
            (p for p in st.session_state.project_list if p["dir"] == st.session_state.current_agent_dir),
            None
        )
        
        if current_project:
            # Project info bar (with rename function)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 📁 Current Project: {current_project['name']}")
            with col2:
                if st.button("✏️ Rename", use_container_width=True):
                    st.session_state.show_rename = True
            
            # Rename dialog
            if st.session_state.get("show_rename", False):
                with st.expander("Rename Project", expanded=True):
                    new_name = st.text_input("New Project Name", value=current_project['name'])
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Confirm", use_container_width=True):
                            if new_name and new_name != current_project['name']:
                                new_dir = rename_project(current_project['dir'], new_name)
                                if new_dir:
                                    st.success("Project renamed successfully!")
                                    st.session_state.current_agent_dir = new_dir
                                    st.session_state.project_list = load_project_directories()
                                    st.session_state.show_rename = False
                                    st.rerun()
                    with col2:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.show_rename = False
                            st.rerun()
            
            # Tabs: Chat, Experiment List, Visualization
            tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Experiment List", "📊 Visualization"])
            
            # Tab 1: Chat
            with tab1:
                st.header(f"Chat with Agent - {current_project['name']}")
                
                # Get or create agent
                agent = get_or_create_agent(st.session_state.current_agent_dir)
                
                if agent:
                    # Get message history for this project
                    messages = st.session_state.agent_messages.get(st.session_state.current_agent_dir, [])
                    
                    # Display message history (exclude the last assistant message if it's being streamed)
                    # This prevents duplication between history display and streaming output
                    display_messages = messages.copy()
                    
                    # If there's a pending response being generated, we'll show it via streaming
                    # So we don't need to show it in history to avoid duplication
                    for message in display_messages:
                        role = message.get("role", "assistant")
                        if role == "tool":
                            with st.chat_message("assistant"):
                                tool_name = message.get("tool_name", "tool")
                                status = message.get("status", "success")
                                st.markdown(f"**🛠️ Tool `{tool_name}` ({status})**")
                                input_payload = message.get("input")
                                if input_payload:
                                    st.caption(f"Input: {input_payload}")
                                output_text = message.get("output")
                                error_text = message.get("error")
                                if output_text:
                                    st.code(output_text)
                                if error_text:
                                    st.error(error_text)
                        kb_results = parse_kb_results([message])
                        if kb_results:
                            render_kb_results(kb_results)
                            continue
                        
                        with st.chat_message(role):
                            if role == "assistant":
                                # 先展示工具调用和检索结果，再展示文本
                                tool_events = message.get("tool_events") or []
                                if tool_events:
                                    with st.expander("🛠️ Tool Calls (this turn)", expanded=True):
                                        for event in tool_events:
                                            tool_name = event.get("tool_name", "tool")
                                            status = event.get("status", "success")
                                            st.markdown(f"**{tool_name}** ({status}) - {event.get('timestamp', '')}")
                                            input_payload = event.get("input")
                                            if input_payload:
                                                st.caption(f"Input: {input_payload}")
                                            output_text = event.get("output")
                                            if output_text:
                                                st.code(output_text)
                                            if event.get("error"):
                                                st.error(event["error"])
                                kb_results = parse_kb_results(tool_events)
                                if kb_results:
                                    render_kb_results(kb_results)
                            # 最后渲染消息文本与阶段
                            st.markdown(message.get("content", ""))
                            if role == "assistant" and message.get("stage"):
                                st.caption(f"Stage: {message['stage'].replace('_', ' ').title()}")
                    
                    # User input
                    if prompt := st.chat_input("Enter your research question or instruction..."):
                        # Add user message
                        messages.append({"role": "user", "content": prompt})
                        st.session_state.agent_messages[st.session_state.current_agent_dir] = messages
                        save_chat_history(st.session_state.current_agent_dir, messages)
                        
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        # Agent response with streaming
                        with st.chat_message("assistant"):
                            accumulated_response = ""
                            
                            try:
                                # Use st.write_stream for real-time streaming output
                                # This will display content as it's generated, including reasoning before tool calls
                                response_generator = agent.process_sync_streaming(prompt)
                                
                                # st.write_stream automatically handles incremental display and accumulation
                                accumulated_response = st.write_stream(response_generator)
                                
                                tool_events = agent.pop_latest_tool_events() if hasattr(agent, "pop_latest_tool_events") else []
                                
                                # Save complete response to history (only once, avoid duplication)
                                if accumulated_response:
                                    # Get current messages to avoid race condition
                                    current_messages = st.session_state.agent_messages.get(st.session_state.current_agent_dir, [])
                                    # Check if this message was already saved (avoid duplicates on rerun)
                                    if (not current_messages or 
                                        current_messages[-1].get("role") != "assistant" or 
                                        current_messages[-1].get("content") != accumulated_response):
                                        stage = infer_workflow_stage(accumulated_response)
                                        assistant_message = {"role": "assistant", "content": accumulated_response}
                                        if stage:
                                            assistant_message["stage"] = stage
                                            st.session_state.agent_stage[st.session_state.current_agent_dir] = stage
                                        if tool_events:
                                            assistant_message["tool_events"] = tool_events
                                        current_messages.append(assistant_message)
                                        st.session_state.agent_messages[st.session_state.current_agent_dir] = current_messages
                                        save_chat_history(st.session_state.current_agent_dir, current_messages)
                            except Exception as e:
                                st.error(f"Error processing message: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.error("Agent initialization failed. Please check environment variables.")
            
            # Tab 2: Experiment List
            with tab2:
                st.header(f"📋 {current_project['name']} - Experiment List")
                
                # Load all experiments in this project
                experiments = load_experiments_in_project(st.session_state.current_agent_dir)
                
                if experiments:
                    # Display experiment list
                    for exp in experiments:
                        with st.expander(f"🔬 {exp['name']} - {exp['rel_path']}", expanded=False):
                            st.markdown(f"**Status:** {render_status_badge(exp['status'])}", unsafe_allow_html=True)
                            st.caption(f"Created: {exp['created_date']}")
                            if exp.get("research_question"):
                                st.text_area("Research Question", exp["research_question"], height=60, disabled=True, key=f"q_{exp['path']}")
                            
                            # Display details button
                            if st.button("View Details", key=f"detail_{exp['path']}"):
                                st.session_state.selected_experiment = exp
                                st.rerun()
                    
                    # If experiment is selected, display details
                    if "selected_experiment" in st.session_state:
                        selected_exp = st.session_state.selected_experiment
                        if selected_exp["path"].startswith(st.session_state.current_agent_dir):
                            st.divider()
                            st.subheader(f"📋 Experiment Details: {selected_exp['name']}")
                            display_manifest(selected_exp["manifest"])
                else:
                    st.info("No experiments in this project yet.")
            
            # Tab 3: Visualization
            with tab3:
                st.header(f"📊 {current_project['name']} - Data Visualization")
                
                # Load all experiments in this project
                experiments = load_experiments_in_project(st.session_state.current_agent_dir)
                
                if experiments:
                    # Select experiment to visualize
                    exp_options = {exp["name"]: exp for exp in experiments}
                    selected_exp_name = st.selectbox("Select Experiment", list(exp_options.keys()))
                    
                    if selected_exp_name:
                        selected_exp = exp_options[selected_exp_name]
                        visualize_metrics(selected_exp["manifest"])
                else:
                    st.info("No experiments in this project yet.")
        else:
            st.warning("Selected project does not exist. Please select again.")
            st.session_state.current_agent_dir = None
    else:
        # When no project is selected, display agent introduction
        st.markdown("---")
        st.markdown("## 🤖 About Agent Economist")
        st.markdown("""
        I'm an AI research assistant specialized in economic system experiments. I help you design controlled experiments, 
        run simulations, and analyze results for research questions about innovation policy, tax effects, labor markets, 
        wealth distribution, and other economic phenomena.
        
        **Workflow**: Share your research question → I design an experiment plan (hypothesis, experiment type, parameters, 
        verification metrics) → Run simulations → Analyze and provide insights.
        
        **Start by selecting a project from the sidebar or entering your research question below!** 💬
        """)
        
        # Initialize agent (for auto-creating projects)
        agent = None
        try:
            from streamlit_agent_wrapper import create_agent_for_streamlit
            agent = create_agent_for_streamlit()
        except Exception as e:
            st.warning(f"Agent initialization failed: {e}")
        
        if agent:
            # Display message history (if there are temporary messages)
            temp_messages = st.session_state.agent_messages.get("__temp__", [])
            for message in temp_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # User input
            if prompt := st.chat_input("Enter your research question or instruction..."):
                # Auto-create project
                project_name = generate_project_name_from_message(prompt)
                project_dir = create_project_auto(project_name, research_question=prompt)
                
                if project_dir:
                    # Set current project
                    st.session_state.current_agent_dir = project_dir
                    st.session_state.project_list = load_project_directories()
                    
                    # Initialize agent and messages
                    st.session_state.agents[project_dir] = agent
                    st.session_state.agent_messages[project_dir] = [
                        {"role": "user", "content": prompt}
                    ]
                    save_chat_history(project_dir, st.session_state.agent_messages[project_dir])
                    save_chat_history(project_dir, st.session_state.agent_messages[project_dir])
                    
                    # Agent response
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        accumulated_response = ""
                        
                        try:
                            # Use st.write_stream for real-time streaming output
                            # This will display content as it's generated, before tool calls
                            response_generator = agent.process_sync_streaming(prompt)
                            accumulated_response = st.write_stream(response_generator)
                            
                            tool_events = agent.pop_latest_tool_events() if hasattr(agent, "pop_latest_tool_events") else []
                            
                            # Save complete response (avoid duplicates)
                            if accumulated_response:
                                messages = st.session_state.agent_messages[project_dir]
                                # Check if this message was already saved (avoid duplicates on rerun)
                                if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != accumulated_response:
                                    stage = infer_workflow_stage(accumulated_response)
                                    assistant_message = {"role": "assistant", "content": accumulated_response}
                                    if stage:
                                        assistant_message["stage"] = stage
                                        st.session_state.agent_stage[project_dir] = stage
                                    if tool_events:
                                        assistant_message["tool_events"] = tool_events
                                    st.session_state.agent_messages[project_dir].append(assistant_message)
                                    save_chat_history(project_dir, st.session_state.agent_messages[project_dir])
                                    save_chat_history(project_dir, st.session_state.agent_messages[project_dir])
                        except Exception as e:
                            st.error(f"Error processing message: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    st.rerun()
                else:
                    st.error("Failed to auto-create project. Please create manually.")


if __name__ == "__main__":
    main()

