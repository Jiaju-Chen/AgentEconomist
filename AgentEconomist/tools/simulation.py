"""
仿真运行工具

提供经济仿真的运行和报告读取功能。
"""

import os
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Optional

from ..core.manifest import load_manifest, save_manifest, ensure_manifest_structure
from ..utils.path import get_project_root, to_absolute, to_relative
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error


def run_simulation(
    manifest_path: str,
    group_name: str = "",
    timeout: int = 3600,
    run_all: bool = False,
) -> str:
    """
    运行经济仿真
    
    IMPORTANT: Use run_all=True to run all configs sequentially (RECOMMENDED to avoid resource conflicts).
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 要运行的配置组名称（如 "control", "treatment"）
        timeout: 超时时间（秒）
        run_all: 如果为 True，串行运行所有配置（推荐，避免 Qdrant 冲突）
    
    Returns:
        XML格式字符串，包含运行结果
    """
    try:
        if run_all:
            return _run_all_simulations(manifest_path, timeout)
        else:
            return _run_single_simulation(manifest_path, group_name, timeout)
            
    except Exception as e:
        return handle_tool_error("run_simulation", e)


def _copy_images_to_public(manifest_path: str, group_name: str) -> None:
    """
    复制仿真生成的图片到 frontend/public 目录
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称
    """
    try:
        manifest = load_manifest(manifest_path)
        if not manifest:
            return
        
        configurations = manifest.get("configurations", {})
        if group_name not in configurations:
            return
        
        config = configurations[group_name]
        experiment_output_dir = config.get("experiment_output_dir", "")
        experiment_name = config.get("experiment_name", "")
        
        if not experiment_output_dir or not experiment_name:
            print(f"[WARN] Missing output_dir or experiment_name for {group_name}")
            return
        
        # 源目录: output/xxx/charts/
        output_path = to_absolute(experiment_output_dir)
        charts_dir = output_path / "charts"
        
        if not charts_dir.exists() or not charts_dir.is_dir():
            print(f"[INFO] No charts directory found: {charts_dir}")
            return
        
        # 目标目录: frontend/public/{experiment_name}/
        project_root = get_project_root()
        public_dir = project_root / "frontend" / "public" / experiment_name
        public_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所有图片
        copied_count = 0
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            for img_file in charts_dir.glob(pattern):
                dest_file = public_dir / img_file.name
                shutil.copy2(img_file, dest_file)
                copied_count += 1
                print(f"[INFO] Copied: {img_file.name} -> {dest_file.relative_to(project_root)}")
        
        if copied_count > 0:
            print(f"[SUCCESS] Copied {copied_count} images to {public_dir.relative_to(project_root)}")
        else:
            print(f"[INFO] No images found in {charts_dir}")
            
    except Exception as e:
        print(f"[ERROR] Failed to copy images for {group_name}: {e}")


def _run_single_simulation(manifest_path: str, group_name: str, timeout: int) -> str:
    """运行单个仿真配置"""
    # Load manifest
    manifest = load_manifest(manifest_path)
    if not manifest:
        return format_tool_output("error", f"Manifest not found: {manifest_path}")
    
    # Get configuration
    configurations = manifest.get("configurations", {})
    if group_name not in configurations:
        return format_tool_output(
            "error",
            f"Configuration group '{group_name}' not found in manifest"
        )
    
    config = configurations[group_name]
    config_path = config.get("path", "")
    
    if not config_path:
        return format_tool_output("error", f"No config path for group '{group_name}'")
    
    # Get absolute path
    config_path_abs = to_absolute(config_path)
    if not config_path_abs.exists():
        return format_tool_output("error", f"Config file not found: {config_path}")
    
    # Get simulation script
    project_root = get_project_root()
    sim_script = project_root / "AgentEconomist" / "run_simulation.sh"
    
    if not sim_script.exists():
        return format_tool_output("error", f"Simulation script not found: {sim_script}")
    
    # Update manifest: mark as running
    _update_run_status(manifest_path, group_name, "running")
    
    # Run simulation
    try:
        result = subprocess.run(
            [str(sim_script), str(config_path_abs)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )
        
        if result.returncode != 0:
            _update_run_status(manifest_path, group_name, "failed", result.stderr[-2000:])
            return format_tool_output(
                "error",
                f"Simulation failed with return code {result.returncode}",
                manifest_path=manifest_path,
                group_name=group_name,
                stderr=result.stderr[-500:] if result.stderr else ""
            )
        
        # Success
        _update_run_status(manifest_path, group_name, "completed")
        
        # 更新 report_file 路径
        _update_report_file(manifest_path, group_name)
        
        # 复制图片到 public 目录
        _copy_images_to_public(manifest_path, group_name)
        
        return format_tool_output(
            "success",
            f"Simulation completed for group '{group_name}'",
            manifest_path=manifest_path,
            group_name=group_name,
            stdout=result.stdout[-500:] if result.stdout else ""
        )
        
    except subprocess.TimeoutExpired:
        _update_run_status(manifest_path, group_name, "failed", f"Timeout after {timeout}s")
        return format_tool_output(
            "error",
            f"Simulation timed out after {timeout} seconds",
            manifest_path=manifest_path,
            group_name=group_name
        )


def _run_all_simulations(manifest_path: str, timeout: int) -> str:
    """串行运行所有仿真配置"""
    manifest = load_manifest(manifest_path)
    if not manifest:
        return format_tool_output("error", f"Manifest not found: {manifest_path}")
    
    configurations = manifest.get("configurations", {})
    if not configurations:
        return format_tool_output("error", "No configurations found in manifest")
    
    # Sort groups: control first
    groups = sorted(configurations.keys(), key=lambda x: (x != "control", x))
    
    results = []
    for group_name in groups:
        result = _run_single_simulation(manifest_path, group_name, timeout)
        results.append(f"[{group_name}] {result}")
        
        # Check if failed
        if "<status>error</status>" in result:
            return format_tool_output(
                "error",
                f"Failed at group '{group_name}', stopping sequential execution",
                manifest_path=manifest_path,
                completed_groups=", ".join(groups[:groups.index(group_name)]),
                failed_group=group_name
            )
    
    return format_tool_output(
        "success",
        f"All {len(groups)} simulations completed successfully",
        manifest_path=manifest_path,
        groups=", ".join(groups)
    )


def _update_run_status(manifest_path: str, group_name: str, status: str, log: str = ""):
    """更新运行状态到 manifest"""
    from datetime import datetime, timezone, timedelta
    
    manifest = ensure_manifest_structure(load_manifest(manifest_path))
    runs = manifest.setdefault("runs", {})
    
    if group_name not in runs:
        runs[group_name] = {
            "config_path": None,
            "parameters_changed": {},
            "status": "planned",
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "report_file": None,
            "run_log": "",
            "key_metrics": {},
            "log_file": None,
            "output_directory": None,
        }
    
    run_entry = runs[group_name]
    now = datetime.now(timezone(timedelta(hours=8)))  # Beijing time
    
    if status == "running":
        run_entry["status"] = "running"
        run_entry["start_time"] = now.isoformat()
        
        # Update experiment-level status
        if manifest["metadata"]["runtime"]["start_time"] is None:
            manifest["metadata"]["runtime"]["start_time"] = now.isoformat()
            manifest["metadata"]["status"] = "running"
            
    elif status in {"completed", "failed"}:
        run_entry["status"] = status
        run_entry["end_time"] = now.isoformat()
        
        if run_entry.get("start_time"):
            start_dt = datetime.fromisoformat(run_entry["start_time"])
            run_entry["duration_seconds"] = (now - start_dt).total_seconds()
        
        if log:
            run_entry["run_log"] = log
        
        # Check if all runs finished
        if all(r.get("status") in {"completed", "failed"} for r in runs.values()):
            manifest["metadata"]["runtime"]["end_time"] = now.isoformat()
            start_time = manifest["metadata"]["runtime"]["start_time"]
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                manifest["metadata"]["runtime"]["duration_seconds"] = (now - start_dt).total_seconds()
            
            if all(r.get("status") == "completed" for r in runs.values()):
                manifest["metadata"]["status"] = "analysis_pending"
            else:
                manifest["metadata"]["status"] = "failed"
    
    save_manifest(manifest_path, manifest)


def _update_report_file(manifest_path: str, group_name: str):
    """
    自动查找并更新 simulation_report 文件路径到 manifest
    
    在仿真完成后，扫描输出目录查找 simulation_report_*.json 文件，
    并将其相对路径更新到 manifest['runs'][group_name]['report_file']
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称
    """
    try:
        manifest = load_manifest(manifest_path)
        if not manifest:
            print(f"Warning: Cannot load manifest {manifest_path}")
            return
        
        # 获取输出目录
        config = manifest.get("configurations", {}).get(group_name, {})
        output_dir = config.get("experiment_output_dir")
        
        if not output_dir:
            # 尝试从 runs 中获取
            runs = manifest.get("runs", {})
            if group_name in runs:
                output_dir = runs[group_name].get("output_directory")
        
        if not output_dir:
            print(f"Warning: No output directory found for group '{group_name}'")
            return
        
        # 转换为绝对路径
        output_path = to_absolute(output_dir)
        
        if not output_path.exists():
            print(f"Warning: Output directory does not exist: {output_path}")
            return
        
        # 查找 simulation_report_*.json 文件
        report_files = list(output_path.glob("simulation_report_*.json"))
        
        if not report_files:
            print(f"Info: No simulation_report_*.json found in {output_path}")
            return
        
        # 如果有多个，选择最新的
        if len(report_files) > 1:
            report_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            print(f"Info: Multiple report files found, using latest: {report_files[0].name}")
        
        report_file = report_files[0]
        
        # 转换为相对于项目根目录的路径
        from ..utils.path import get_project_root
        project_root = get_project_root()
        
        try:
            report_file_relative = report_file.relative_to(project_root)
        except ValueError:
            # 如果文件不在项目根目录下，使用绝对路径
            report_file_relative = report_file
        
        # 更新 manifest
        runs = manifest.setdefault("runs", {})
        if group_name not in runs:
            runs[group_name] = {}
        
        runs[group_name]["report_file"] = str(report_file_relative)
        
        # 保存
        save_manifest(manifest_path, manifest)
        print(f"[SUCCESS] Updated report_file for '{group_name}': {report_file_relative}")
        
    except Exception as e:
        print(f"Warning: Failed to update report_file for '{group_name}': {e}")


def read_simulation_report(
    manifest_path: str,
    group_name: Optional[str] = None
) -> str:
    """
    读取仿真报告（simulation_report_*.json）
    
    此工具读取实验输出目录中的 simulation_report_{timestamp}.json 文件，
    该文件包含仿真的快速摘要（与 economic_metrics_history.json 内容类似）。
    
    注意：如果需要详细的经济指标分析，建议使用 analyze_experiment_directory，
    它会从 data/ 目录下提取更全面的数据：
    - economic_metrics_history.json
    - household_monthly_metrics.json
    - firm_monthly_metrics.json
    - performance_metrics.json
    - innovation_configs.json
    
    Args:
        manifest_path: manifest.yaml 路径
        group_name: 配置组名称（可选，默认读取所有组）
    
    Returns:
        XML格式字符串，包含 simulation_report.json 的内容摘要
    """
    try:
        manifest = load_manifest(manifest_path)
        if not manifest:
            return format_tool_output("error", f"Manifest not found: {manifest_path}")
        
        runs = manifest.get("runs", {})
        
        if group_name:
            if group_name not in runs:
                return format_tool_output("error", f"Group '{group_name}' not found in runs")
            
            run_entry = runs[group_name]
            report_file = run_entry.get("report_file")
            
            if not report_file:
                return format_tool_output("error", f"No report file for group '{group_name}'")
            
            # Read report
            report_path = to_absolute(report_file)
            if not report_path.exists():
                return format_tool_output("error", f"Report file not found: {report_file}")
            
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
            
            return format_tool_output(
                "success",
                f"Report for group '{group_name}'",
                manifest_path=manifest_path,
                group_name=group_name,
                report=report_content[:2000]  # Limit size
            )
        else:
            # Read all reports
            reports = {}
            for gname, run_entry in runs.items():
                report_file = run_entry.get("report_file")
                if report_file:
                    report_path = to_absolute(report_file)
                    if report_path.exists():
                        with open(report_path, "r", encoding="utf-8") as f:
                            reports[gname] = f.read()[:1000]
            
            return format_tool_output(
                "success",
                f"Found {len(reports)} reports",
                manifest_path=manifest_path,
                groups=", ".join(reports.keys()),
                reports=str(reports)[:2000]
            )
            
    except Exception as e:
        return handle_tool_error("read_simulation_report", e)
