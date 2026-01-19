"""
YAML 操作模块

提供 YAML 配置文件的读取、修改和生成功能。
"""

import os
import yaml
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple
from pathlib import Path

from ..core.manifest import load_manifest, save_manifest, ensure_manifest_structure
from ..utils.path import get_project_root, to_absolute, to_relative
from ..utils.format import format_tool_output


def create_config_from_template(
    manifest_path: str,
    group_name: str,
    parameter_changes: Dict[str, Any]
) -> str:
    """
    从模板创建配置文件并修改参数
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称
        parameter_changes: 参数修改字典（扁平化格式，如 {"system_scale.num_households": 200}）
    
    Returns:
        XML格式字符串
    """
    try:
        # Load default template
        project_root = get_project_root()
        template_path = project_root / "default.yaml"
        
        if not template_path.exists():
            return format_tool_output("error", f"Template not found: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        
        # Apply parameter changes
        for param_path, value in parameter_changes.items():
            tokens = _parse_path(param_path)
            config_data = _set_value(config_data, tokens, value, create_missing=True)
        
        # Ensure experiment metadata
        experiment_name, experiment_output_dir = _ensure_experiment_metadata(config_data)
        
        # Save config file
        manifest_abs = to_absolute(manifest_path)
        experiment_dir = manifest_abs.parent
        config_file_path = experiment_dir / f"{group_name}.yaml"
        
        with open(config_file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, allow_unicode=True, sort_keys=False)
        
        # Update manifest
        manifest = ensure_manifest_structure(load_manifest(manifest_path))
        configurations = manifest.setdefault("configurations", {})
        runs = manifest.setdefault("runs", {})
        
        configurations[group_name] = {
            "path": str(config_file_path),
            "parameters_changed": parameter_changes,
            "experiment_name": experiment_name,
            "experiment_output_dir": experiment_output_dir,
        }
        
        runs[group_name] = {
            "config_path": str(config_file_path),
            "parameters_changed": parameter_changes,
            "status": "planned",
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "report_file": None,
            "run_log": "",
            "key_metrics": {},
            "log_file": str(experiment_dir / f"{group_name}.log"),
            "output_directory": experiment_output_dir,
        }
        
        save_manifest(manifest_path, manifest)
        
        return format_tool_output(
            "success",
            f"Configuration created for group '{group_name}'",
            manifest_path=manifest_path,
            group_name=group_name,
            config_path=to_relative(str(config_file_path)),
            experiment_name=experiment_name
        )
        
    except Exception as e:
        return format_tool_output("error", f"Failed to create config: {str(e)}")


def _parse_path(path: str) -> List[Any]:
    """解析路径字符串为 tokens 列表"""
    tokens: List[Any] = []
    if not path:
        return tokens
    
    parts = path.split(".")
    for part in parts:
        # Check for array index
        match = re.match(r"(\w+)\[(\d+)\]", part)
        if match:
            tokens.append(match.group(1))
            tokens.append(int(match.group(2)))
        else:
            tokens.append(part)
    
    return tokens


def _set_value(data: Any, tokens: List[Any], value: Any, create_missing: bool) -> Any:
    """在嵌套数据结构中设置值"""
    if not tokens:
        return value
    
    current = data
    for i, token in enumerate(tokens[:-1]):
        next_token = tokens[i + 1]
        
        if isinstance(token, int):
            # Array index
            if not isinstance(current, list):
                raise TypeError(f"Expected list at {token}")
            while len(current) <= token:
                if not create_missing:
                    raise KeyError(f"Index {token} out of range")
                current.append({} if isinstance(next_token, str) else [])
            current = current[token]
        else:
            # Dict key
            if not isinstance(current, dict):
                raise TypeError(f"Expected dict at {token}")
            if token not in current:
                if not create_missing:
                    raise KeyError(f"Key '{token}' not found")
                current[token] = {} if isinstance(next_token, str) else []
            current = current[token]
    
    # Set final value
    last_token = tokens[-1]
    if isinstance(last_token, int):
        if not isinstance(current, list):
            raise TypeError(f"Expected list at {last_token}")
        while len(current) <= last_token:
            if not create_missing:
                raise KeyError(f"Index {last_token} out of range")
            current.append(None)
        current[last_token] = value
    else:
        if not isinstance(current, dict):
            raise TypeError(f"Expected dict at {last_token}")
        current[last_token] = value
    
    return data


def _ensure_experiment_metadata(data: Dict) -> Tuple[str, str]:
    """确保实验元数据存在"""
    experiment_name = data.get("experiment_name")
    experiment_output_dir = data.get("experiment_output_dir")
    
    if experiment_name and experiment_output_dir:
        return experiment_name, experiment_output_dir
    
    # Extract system scale info
    system_scale = data.get("system_scale", {})
    num_households = system_scale.get("num_households", 0)
    num_iterations = system_scale.get("num_iterations", 0)
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment_name or f"exp_{num_households}h_{num_iterations}m_{timestamp}"
    
    # Generate output directory
    project_root = get_project_root()
    experiment_output_dir = experiment_output_dir or str(project_root / "output" / experiment_name)
    
    data["experiment_name"] = experiment_name
    data["experiment_output_dir"] = experiment_output_dir
    
    return experiment_name, experiment_output_dir
