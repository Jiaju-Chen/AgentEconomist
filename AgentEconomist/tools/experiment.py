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

from ..core.manifest import save_manifest, ensure_manifest_structure, load_manifest
from ..utils.path import get_project_root, to_relative
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error


def init_manifest() -> str:
    """
    初始化实验目录和最小化 manifest。
    在工作流最开始调用，立即生成 manifest_path。
    
    Returns:
        XML格式字符串，包含 manifest_path
    """
    try:
        # 生成基于时间戳的目录名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir_name = f"experiment_{timestamp}"
        project_root = get_project_root()
        experiment_dir = project_root / "experiment_files" / experiment_dir_name
        
        # 创建目录
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建最小化 manifest 结构
        manifest_data = {
            "experiment_info": {
                "name": "",
                "description": "",
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "author": "DesignAgent",
                "tags": [],
                "notes": "",
                "directory": str(experiment_dir),
            },
            "metadata": {
                "research_question": "",
                "hypothesis": "",
                "expected_outcome": "",
                "status": "planned",
                "runtime": {
                    "start_time": None,
                    "end_time": None,
                    "duration_seconds": None,
                },
                "knowledge_base": []
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
        
        # 保存 manifest
        manifest_path = experiment_dir / "manifest.yaml"
        save_manifest(str(manifest_path), manifest_data)
        
        # 返回相对路径
        manifest_path_rel = to_relative(str(manifest_path))
        
        return format_tool_output(
            "success",
            f"Experiment directory created: {experiment_dir_name}",
            manifest_path=manifest_path_rel,
            experiment_directory=to_relative(str(experiment_dir))
        )
        
    except Exception as e:
        return handle_tool_error("init_manifest", e)


def update_experiment_metadata(
    manifest_path: str,
    name: str = "",
    description: str = "",
    research_question: str = "",
    hypothesis: str = "",
    expected_outcome: str = "",
    tags: Optional[str] = None
) -> str:
    """
    更新实验 metadata 信息。
    可以多次调用来更新不同字段。
    
    注意：不包括 status 参数，status 由 run_simulation 自动管理，避免冲突。
    
    Args:
        manifest_path: manifest.yaml 路径
        name: 实验名称
        description: 实验描述
        research_question: 研究问题
        hypothesis: 研究假设
        expected_outcome: 预期结果
        tags: 标签列表（逗号分隔字符串）
    
    Returns:
        XML格式字符串
    """
    try:
        manifest = ensure_manifest_structure(load_manifest(manifest_path))
        
        # 更新 experiment_info
        if name:
            manifest["experiment_info"]["name"] = name
        if description:
            manifest["experiment_info"]["description"] = description
        if tags:
            tags_list = [t.strip() for t in tags.split(",") if t.strip()]
            manifest["experiment_info"]["tags"] = tags_list
        
        # 更新 metadata（不包括 status - 由 run_simulation 管理）
        if research_question:
            manifest["metadata"]["research_question"] = research_question
        if hypothesis:
            manifest["metadata"]["hypothesis"] = hypothesis
        if expected_outcome:
            manifest["metadata"]["expected_outcome"] = expected_outcome
        
        save_manifest(manifest_path, manifest)
        
        # 构建更新字段列表（仅显示非空字段）
        updated_fields = {}
        if name:
            updated_fields["name"] = name
        if description:
            updated_fields["description"] = description
        if research_question:
            updated_fields["research_question"] = research_question
        if hypothesis:
            updated_fields["hypothesis"] = hypothesis
        if expected_outcome:
            updated_fields["expected_outcome"] = expected_outcome
        if tags:
            updated_fields["tags"] = tags
        
        return format_tool_output(
            "success",
            "Experiment metadata updated",
            manifest_path=manifest_path,
            updated_fields=updated_fields
        )
        
    except Exception as e:
        return handle_tool_error("update_experiment_metadata", e)


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


def modify_yaml_parameters(
    manifest_path: str,
    group_names: str,
    parameter_changes: Union[str, dict]
) -> str:
    """
    批量修改多个配置组的参数。
    确保所有配置同步更新，避免配置丢失问题。
    
    Args:
        manifest_path: manifest.yaml 路径
        group_names: 配置组名称（逗号分隔，如 "control,treatment"）
        parameter_changes: 参数修改（JSON字符串或字典）
    
    Returns:
        XML格式字符串
    
    Example:
        modify_yaml_parameters(
            manifest_path="experiment_files/exp_123/manifest.yaml",
            group_names="control,treatment",
            parameter_changes='{"system_scale.num_iterations": 2}'
        )
    """
    try:
        from ..core.yaml_ops import modify_existing_yaml
        
        # Parse parameters
        if isinstance(parameter_changes, str):
            try:
                params_dict = json.loads(parameter_changes)
            except json.JSONDecodeError as e:
                return handle_tool_error("modify_yaml_parameters", 
                    ValueError(f"Invalid JSON in parameter_changes: {e}"))
        else:
            params_dict = parameter_changes
        
        # Parse group names
        groups = [g.strip() for g in group_names.split(",") if g.strip()]
        if not groups:
            return handle_tool_error("modify_yaml_parameters", 
                ValueError("No group names provided"))
        
        # Load manifest
        manifest = ensure_manifest_structure(load_manifest(manifest_path))
        configurations = manifest.get("configurations", {})
        
        # Check that all groups exist
        missing_groups = [g for g in groups if g not in configurations]
        if missing_groups:
            return handle_tool_error("modify_yaml_parameters",
                ValueError(f"Groups not found in manifest: {missing_groups}"))
        
        # Modify each configuration file
        modified_configs = []
        for group_name in groups:
            config_info = configurations[group_name]
            config_path = config_info.get("path", "")
            
            if not config_path:
                continue
            
            # Modify the YAML file
            modify_existing_yaml(config_path, params_dict)
            
            # Update manifest with new parameters_changed
            existing_params = config_info.get("parameters_changed", {})
            existing_params.update(params_dict)
            config_info["parameters_changed"] = existing_params
            
            modified_configs.append({
                "group": group_name,
                "path": to_relative(config_path),
                "updated_params": params_dict
            })
        
        # Save manifest
        save_manifest(manifest_path, manifest)
        
        return format_tool_output(
            "success",
            f"Modified {len(modified_configs)} configuration(s): {', '.join(groups)}",
            manifest_path=manifest_path,
            modified_configs=modified_configs,
            parameter_changes=params_dict
        )
        
    except Exception as e:
        return handle_tool_error("modify_yaml_parameters", e)
