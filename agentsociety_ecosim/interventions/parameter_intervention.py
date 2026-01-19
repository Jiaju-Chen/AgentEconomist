"""
参数干预实现
"""
from typing import Dict, Any
from .base import Intervention
import logging

logger = logging.getLogger(__name__)


class ParameterInterventionExecutor:
    """参数干预执行器"""
    
    @staticmethod
    async def execute(intervention: Intervention, simulation, parameter_manager) -> Dict[str, Any]:
        """
        执行参数干预
        
        Args:
            intervention: 干预对象
            simulation: EconomicSimulation 实例
            parameter_manager: ParameterManager 实例
            
        Returns:
            执行结果字典
        """
        try:
            parameters = intervention.parameters
            results = {}
            
            logger.info(f"执行参数干预 {intervention.intervention_id}: {intervention.description}")
            
            # 遍历所有要修改的参数
            for param_name, new_value in parameters.items():
                # ✨ 关键修复：直接修改仿真实例的config，而不是parameter_manager的config
                old_value = getattr(simulation.config, param_name, None)
                
                # 验证参数（使用parameter_manager验证）
                if parameter_manager:
                    validation_result = parameter_manager.set_parameter(
                        param_name, 
                        new_value, 
                        validate=True
                    )
                    
                    if not validation_result.success:
                        results[param_name] = {
                            "success": False,
                            "error": validation_result.errors
                        }
                        logger.error(f"  参数 {param_name} 验证失败: {validation_result.errors}")
                        continue
                
                # ✨ 直接设置到仿真实例的config
                setattr(simulation.config, param_name, new_value)
                
                results[param_name] = {
                    "success": True,
                    "old_value": old_value,
                    "new_value": new_value
                }
                logger.info(f"  参数 {param_name}: {old_value} → {new_value} (已应用到仿真)")
            
            # 如果需要同步到Ray actors (例如税率变化需要同步到economic_center)
            await ParameterInterventionExecutor._sync_to_ray_actors(
                simulation, 
                parameters
            )
            
            return {
                "success": True,
                "parameters_changed": len([r for r in results.values() if r.get("success")]),
                "details": results
            }
            
        except Exception as e:
            logger.error(f"参数干预执行失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def _sync_to_ray_actors(simulation, parameters: Dict[str, Any]):
        """
        同步参数到Ray actors和仿真对象
        
        注意：大多数参数是在仿真的config对象中读取的，不需要同步。
        但有些参数存储在对象内部（如家庭的税率），需要显式同步。
        """
        try:
            sync_count = 0
            
            # 1. 同步税率到家庭对象
            household_tax_updated = False
            if "income_tax_rate" in parameters or "vat_rate" in parameters:
                if hasattr(simulation, 'households') and simulation.households:
                    income_tax = parameters.get("income_tax_rate")
                    vat_rate = parameters.get("vat_rate")
                    
                    for household in simulation.households:
                        if income_tax is not None:
                            household.income_tax_rate = income_tax
                        if vat_rate is not None:
                            household.vat_rate = vat_rate
                    
                    household_tax_updated = True
                    sync_count += len(simulation.households)
                    logger.info(f"✅ 已同步税率到 {len(simulation.households)} 个家庭对象")
                    if income_tax is not None:
                        logger.info(f"   个人所得税率: → {income_tax}")
                    if vat_rate is not None:
                        logger.info(f"   增值税率: → {vat_rate}")
            
            # 2. 同步企业所得税率到企业对象（如果企业对象也存储了税率）
            if "corporate_tax_rate" in parameters:
                if hasattr(simulation, 'firms') and simulation.firms:
                    corporate_tax = parameters["corporate_tax_rate"]
                    # 检查企业对象是否有 corporate_tax_rate 属性
                    firms_updated = 0
                    for firm in simulation.firms:
                        if hasattr(firm, 'corporate_tax_rate'):
                            firm.corporate_tax_rate = corporate_tax
                            firms_updated += 1
                    
                    if firms_updated > 0:
                        sync_count += firms_updated
                        logger.info(f"✅ 已同步企业所得税率到 {firms_updated} 个企业对象: {corporate_tax}")
            
            # 3. 税率参数 - 通知EconomicCenter（如果有更新方法）
            tax_params = {}
            if "income_tax_rate" in parameters:
                tax_params["income_tax_rate"] = parameters["income_tax_rate"]
            if "vat_rate" in parameters:
                tax_params["vat_rate"] = parameters["vat_rate"]
            if "corporate_tax_rate" in parameters:
                tax_params["corporate_tax_rate"] = parameters["corporate_tax_rate"]
            
            if tax_params:
                logger.info(f"📋 税率参数已更新到simulation.config: {tax_params}")
                if hasattr(simulation, 'economic_center'):
                    await simulation.economic_center.update_tax_rates.remote(**tax_params)
            # 4. 其他需要同步的参数
            # 大多数参数（如labor_productivity, profit_to_production_ratio等）
            # 都是从simulation.config直接读取，不需要额外同步
            
            if sync_count > 0:
                logger.info(f"✅ 参数同步完成：已更新 {sync_count} 个对象")
            else:
                logger.info("✅ 参数已应用到simulation.config，后续操作将使用新值")
            
        except Exception as e:
            logger.warning(f"参数同步失败: {e}", exc_info=True)

