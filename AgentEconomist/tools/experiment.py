"""
实验管理工具

提供实验清单（manifest）的初始化和配置文件生成功能。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union
import json

from ..core.manifest import save_manifest, ensure_manifest_structure
from ..utils.path import get_project_root, to_relative
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error


def init_experiment_manifest(
    experiment_name: str,
    description: str,
    research_question: str = "",
    hypothesis: str = "",
    expected_outcome: str = "",
    tags: Optional[str] = None
) -> str:
    """
    初始化实验清单（manifest.yaml）
    
    Args:
        experiment_name: 实验名称
        description: 实验描述
        research_question: 研究问题
        hypothesis: 研究假设
        expected_outcome: 预期结果
        tags: 标签列表（逗号分隔字符串）
    
    Returns:
        XML格式字符串，包含 manifest_path
    """
    try:
        # Convert tags string to list
        tags_list = [t.strip() for t in tags.split(",")] if tags else []
        
        # Generate experiment directory path based on name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir_name = f"{timestamp}_{experiment_name}"
        project_root = get_project_root()
        experiment_dir = project_root / "experiment_files" / experiment_dir_name
        
        # Create directory
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create manifest data
        manifest_data = {
            "experiment_info": {
                "name": experiment_name,
                "description": description,
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "author": "DesignAgent",
                "tags": tags_list,
                "notes": "",
                "directory": str(experiment_dir),
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
                "intervention_type": "none",
                "intervention_parameters": {},
            },
            "configurations": {},
            "runs": {},
            "results_summary": {
                "comparison_status": "pending",
                "conclusion": "",
                "insights": [],
            },
        }
        
        # Ensure complete structure
        manifest_data = ensure_manifest_structure(manifest_data)
        
        # Save manifest
        manifest_path = experiment_dir / "manifest.yaml"
        save_manifest(str(manifest_path), manifest_data)
        
        # Return relative path
        manifest_path_rel = to_relative(str(manifest_path))
        
        return format_tool_output(
            "success",
            f"Experiment manifest created: {experiment_name}",
            manifest_path=manifest_path_rel,
            experiment_name=experiment_name,
            experiment_directory=to_relative(str(experiment_dir))
        )
        
    except Exception as e:
        return handle_tool_error("init_experiment_manifest", e)


def create_yaml_from_template(
    manifest_path: str,
    group_name: str,
    parameters: Union[str, dict]
) -> str:
    """
    从模板创建配置文件并修改参数
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称（如 "control", "treatment"）
        parameters: 参数修改（JSON字符串或字典，如 '{"system_scale.num_households": 200}'）
    
    Returns:
        XML格式字符串，包含配置文件路径
    """
    try:
        # Parse parameters
        if isinstance(parameters, str):
            try:
                params_dict = json.loads(parameters)
            except json.JSONDecodeError as e:
                return format_tool_output(
                    "error",
                    f"Invalid JSON format in parameters: {str(e)}"
                )
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            return format_tool_output(
                "error",
                f"Parameters must be a JSON string or dict, got {type(parameters).__name__}"
            )
        
        # Import yaml_ops (will be created next)
        from ..core import yaml_ops
        
        # Call yaml operations
        result = yaml_ops.create_config_from_template(
            manifest_path=manifest_path,
            group_name=group_name,
            parameter_changes=params_dict
        )
        
        return result
        
    except Exception as e:
        return handle_tool_error("create_yaml_from_template", e)
