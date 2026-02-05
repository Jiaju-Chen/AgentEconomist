from typing import List, Optional, Dict, Tuple, Any
from numpy import inf
import ray
from .model import Job, MatchedJob, LaborHour, JobApplication
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(name="jobmarket")
import json
import os
import tiktoken

def calculate_tokens(text: str) -> int:
    """计算文本的token数量"""
    try:
        encoding = tiktoken.encoding_for_model('gpt-4')
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token计算失败: {e}")
        return len(text.split()) * 1.3  # 粗略估算

@ray.remote(num_cpus=8)
class LaborMarket:
    def __init__(self):
        self.job_postings: List[Job] = []
        self.household_to_company: Dict[str, List[str]] = {}  # Tracks household ID to company ID mapping
        self.matched_jobs: List[MatchedJob] = []
        # 新增：存储所有工作申请，按job_id分组
        self.job_applications: Dict[str, List[JobApplication]] = {}  # job_id -> List[JobApplication]
        # 新增：存储备选候选人，按job_id分组
        self.backup_candidates: Dict[str, List[Dict]] = {}  # job_id -> List[backup_candidate_info]
        from openai import AsyncOpenAI
        # 使用环境变量获取API key - 改为异步客户端实现真正并发
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
            timeout=60.0  # 设置60秒超时
        )

        logger.info("LaborMarket initialized.")
    def publish_job(self, jobs: Job):
        self.job_postings.extend(jobs)

    def query_jobs(self, company_id):
        return [job for job in self.job_postings if job.company_id == company_id]
    
    def query_matched_jobs(self):
        return self.matched_jobs
    
    def add_job_position(self, company_id, job: Job):
        """
        Adds a job position to the market for a specific company.
        If the job already exists, it increments the available positions.
        """
        for j in self.job_postings:
            if j.company_id == company_id and j.SOC == job.SOC:
                j.positions_available += 1
                return

    async def align_job(self, household_id: str, job: Job, lh_type: str):
        """
        Aligns a job with a household, reducing the available positions.
        """
        for j in self.job_postings:
            if j.SOC == job.SOC and j.company_id == job.company_id and j.positions_available > 0:
                # j.is_valid = False  # Mark job as no longer available
                j.household_id = household_id  # Record the household that applied
                j.positions_available -= 1  # Decrease the number of available positions
                if j.positions_available <= 0:
                    j.is_valid = False
                # logger.info(f"Job {j.title} aligned with household {household_id}.")
                # print(f"Job {j.title} aligned with household {household_id}.")
                self.matched_jobs.append(
                    MatchedJob.create(job=j, average_wage=j.wage_per_hour, household_id=household_id, lh_type=lh_type, company_id=job.company_id)
                )
                return  j
        # logger.warning(f"Job {job.title} id {job.SOC} not found for alignment with household {household_id}.")
        return False
        
    def get_open_jobs(self) -> List[Job]:
        """Returns a list of all jobs with available positions."""
        return [job for job in self.job_postings if job.is_valid and job.positions_available > 0]

    def get_unemployment_statistics(self) -> Dict[str, Any]:
        """
        获取当前的失业统计数据
        
        Returns:
            Dict包含失业统计信息
        """
        try:
            # 统计已匹配的工作
            total_matched = len(self.matched_jobs)
            
            # 统计当前开放的工作岗位
            total_open_positions = sum(job.positions_available for job in self.job_postings if job.is_valid)
            
            # 统计总的工作申请数量
            total_applications = sum(len(apps) for apps in self.job_applications.values())
            
            # 计算失业率（简化计算）
            # 假设申请数量反映了求职者数量，已匹配工作反映了就业数量
            unemployed_count = max(0, total_applications - total_matched)
            unemployment_rate = unemployed_count / max(1, total_applications) if total_applications > 0 else 0
            
            return {
                "total_labor_force_unemployed": unemployed_count,
                "total_labor_force_available": total_applications,
                "total_labor_force_employed": total_matched,
                "unemployment_rate": unemployment_rate,
                "total_open_positions": total_open_positions,
                "total_applications": total_applications
            }
        except Exception as e:
            logger.error(f"获取失业统计数据失败: {e}")
            return {
                "total_labor_force_unemployed": 0,
                "total_labor_force_available": 0,
                "total_labor_force_employed": 0,
                "unemployment_rate": 0,
                "total_open_positions": 0,
                "total_applications": 0
            }

    def apply_for_job(self, household_id: str, company_id: str,  hours_household_can_work: float) -> Optional[tuple[Job, float, float]]:
        """
        A household applies for a specific job.
        If successful, returns a tuple: (job_object, hours_assigned, total_wage_for_period).
        Otherwise, returns None.
        """
        job = next((j for j in self.job_postings if j.company_id == company_id and j.positions_available > 0 ), None)

        if not job:
            # print(f"LaborMarket: Job ID {job_id} not found for application by {household_id}.") # Optional
            return None
        
        if job.positions_available <= 0:
            # print(f"LaborMarket: Job '{job.title}' (ID: {job_id}) has no open positions for {household_id}.") # Optional
            return None

        # Determine actual hours to assign: minimum of what the job offers and what household can work.
        # For simplicity, we assume the household applies for the job's standard hours,
        # or the job is flexible up to job.hours_per_period.
        # Here, we'll assume the job dictates the hours, and household must meet them.
        # A more complex model could allow negotiation or partial hour fulfillment.
        
        if hours_household_can_work < job.hours_per_period:
            # print(f"LaborMarket: Household {household_id} cannot meet required hours ({job.hours_per_period}) for job '{job.title}'. Can only work {hours_household_can_work}.") # Optional
            # Depending on policy, could reject or offer fewer hours if job is divisible.
            # For now, let's assume the job needs its specified hours.
            return None # Or assign min(hours_household_can_work, job.hours_per_period) if job is flexible

        assigned_hours = job.hours_per_period # Household works the job's standard hours
        
        job.positions_available -= 1
        total_wage_for_period = assigned_hours * job.wage_per_hour

        # Record the company ID for this household
        if household_id not in self.household_to_company:
            self.household_to_company[household_id] = []
        if company_id not in self.household_to_company[household_id]:
            self.household_to_company[household_id].append(company_id)  
        
        # print(f"LaborMarket: Household {household_id} successfully hired for '{job.title}' (ID: {job_id}) for {assigned_hours} hrs at ${job.wage_per_hour}/hr. Wage: ${total_wage_for_period:.2f}. Positions left: {job.positions_available}") # Optional
        return job, assigned_hours, total_wage_for_period

    def get_company_for_household(self, household_id: str) ->  Optional[List[str]]:
        """
        Returns the company ID that the household has applied to, or None if no successful application exists.
        """
        return self.household_to_company.get(household_id, None)

    async def match_jobs(self, labor_hour: LaborHour) -> List[Job]: 
        """
        Matches labor hours with available jobs.
        Returns top 3 best matching jobs sorted by matching loss (best match first).
        """
        job_losses = []
        
        for job in self.job_postings:
            if job.is_valid and job.positions_available > 0:
                required_profile = [job.required_skills, job.required_abilities]
                worker_profile = [labor_hour.skill_profile, labor_hour.ability_profile]
                loss = self._compute_matching_loss(worker_profile, required_profile)
                
                # 只考虑损失小于阈值的工作（提高阈值以允许更多匹配）
                if loss < 10000:
                    job_losses.append((job, loss))
        
        # 按损失排序（损失越小越好）
        job_losses.sort(key=lambda x: x[1])
        
        # 返回前3个最佳匹配的工作
        top_jobs = [job for job, loss in job_losses[:3]]
        return top_jobs
    
    def _compute_matching_loss(self, worker_profile: list, required_profile: list) -> float:
        total_loss = 0.0

        for i in range(len(worker_profile)):
            for skill, req in required_profile[i].items():
                mean = req.get('mean')
                std = req.get('std')
                importance = req.get('importance', 1.0)

                # 安全过滤
                if importance is None or importance <= 0:
                    continue
                if std is None or std <= 0:
                    continue
                if mean is None:
                    continue

                # worker 侧没有这个技能，就用 0
                worker_value = worker_profile[i].get(skill, 0.0)

                # 计算标准化偏离
                distance = (worker_value - mean) / std
                
                # 如果工人技能超过要求，给予奖励（降低loss）
                if distance > 0:  # 工人技能超过要求
                    # 使用较小的惩罚，甚至给予奖励
                    loss = importance * (distance ** 2) * 0.1  # 大幅降低超技能的惩罚
                else:  # 工人技能不足
                    loss = importance * (distance ** 2)

                total_loss += loss

        return total_loss
        
    async def firm_handle_job_matching(self, job: Job, labor_hour: LaborHour):
        """
        Firm handles job matching.
        """
        match_score = self.calculate_skill_match_score(labor_hour.skill_profile, labor_hour.ability_profile, job.required_skills, job.required_abilities)
        return await self.llm_set_wage(job, labor_hour.skill_profile, labor_hour.ability_profile, job.wage_per_hour, match_score)
    
    def calculate_skill_match_score(self, worker_skills, worker_abilities, job_skills, job_abilities):
        """
        计算工人技能与工作要求的匹配分数
        返回 0-1 之间的分数，1表示完美匹配
        """
        total_score = 0
        total_weight = 0
        
        # 计算技能匹配分数
        for skill_name, skill_req in job_skills.items():
            if skill_name in worker_skills:
                required_mean = skill_req.get('mean', 50)
                required_std = skill_req.get('std', 10)
                importance = skill_req.get('importance', 1.0)
                
                worker_value = worker_skills[skill_name]
                
                # 计算标准化距离
                if required_std > 0:
                    distance = abs(worker_value - required_mean) / required_std
                    # 转换为0-1分数，距离越小分数越高
                    skill_score = max(0, 1 - distance / 3)  # 3个标准差外为0分
                else:
                    skill_score = 1.0 if worker_value == required_mean else 0.5
                # print(f"Skill {skill_name} score: {skill_score}")
                total_score += skill_score * importance
                total_weight += importance
        
        # 计算能力匹配分数
        for ability_name, ability_req in job_abilities.items():
            if ability_name in worker_abilities:
                required_mean = ability_req.get('mean', 50)
                required_std = ability_req.get('std', 10)
                importance = ability_req.get('importance', 1.0)
                
                worker_value = worker_abilities[ability_name]
                
                if required_std > 0:
                    distance = abs(worker_value - required_mean) / required_std
                    ability_score = max(0, 1 - distance / 3)
                else:
                    ability_score = 1.0 if worker_value == required_mean else 0.5
                # print(f"Ability {ability_name} score: {ability_score}")
                total_score += ability_score * importance
                total_weight += importance
        
        return total_score / total_weight if total_weight > 0 else 0

    async def llm_set_wage(self, job_info, worker_skills, worker_abilities, base_wage, match_score):
        """
        使用LLM根据技能匹配度设定合理工资 (优化版本)
        """
        
        # 优化技能/能力展示 - 应用与招聘决策相同的选择逻辑
        def compress_job_requirements(req_dict, max_items=5):
            """压缩职位要求展示"""
            if not req_dict:
                return "None"
            items = list(req_dict.items())[:max_items]
            return ", ".join([f"{k}({v.get('mean', 50)})" for k, v in items])
        
        def select_key_worker_skills(worker_skills, job_skills, max_items=5):
            """选择关键的工人技能进行展示"""
            if not worker_skills:
                return "None"
            if not job_skills:
                return ", ".join([f"{k}:{v}" for k, v in sorted(worker_skills.items(), key=lambda x: x[1], reverse=True)[:max_items]])
            
            # 优先展示职位要求的技能
            job_required = [(skill, worker_skills.get(skill, 0)) for skill in job_skills.keys()]
            job_required.sort(key=lambda x: job_skills.get(x[0], {}).get('importance', 1.0), reverse=True)
            
            # 补充工人的高值技能
            other_skills = [(k, v) for k, v in worker_skills.items() if k not in job_skills]
            other_skills.sort(key=lambda x: x[1], reverse=True)
            
            result = job_required[:max_items]
            remaining = max_items - len(result)
            if remaining > 0:
                result.extend(other_skills[:remaining])
            
            return ", ".join([f"{k}:{v}" for k, v in result[:max_items]])
        
        def select_key_worker_abilities(worker_abilities, job_abilities, max_items=3):
            """选择关键的工人能力进行展示"""
            if not worker_abilities:
                return "None"
            if not job_abilities:
                return ", ".join([f"{k}:{v}" for k, v in sorted(worker_abilities.items(), key=lambda x: x[1], reverse=True)[:max_items]])
            
            # 优先展示职位要求的能力
            job_required = [(ability, worker_abilities.get(ability, 0)) for ability in job_abilities.keys()]
            job_required.sort(key=lambda x: job_abilities.get(x[0], {}).get('importance', 1.0), reverse=True)
            
            # 补充工人的高值能力
            other_abilities = [(k, v) for k, v in worker_abilities.items() if k not in job_abilities]
            other_abilities.sort(key=lambda x: x[1], reverse=True)
            
            result = job_required[:max_items]
            remaining = max_items - len(result)
            if remaining > 0:
                result.extend(other_abilities[:remaining])
            
            return ", ".join([f"{k}:{v}" for k, v in result[:max_items]])
        
        # 构建优化的英文prompt
        prompt = f"""=== Wage Setting Analysis ===
Position: {job_info['Title']} | Market: ${base_wage:.2f}/h | Match Score: {match_score:.2f}

=== Job Requirements ===
Skills: {compress_job_requirements(job_info.get('skills', {}), 5)}
Abilities: {compress_job_requirements(job_info.get('abilities', {}), 3)}

=== Candidate Profile ===
Skills: {select_key_worker_skills(worker_skills or {}, job_info.get('skills', {}), 5)}
Abilities: {select_key_worker_abilities(worker_abilities or {}, job_info.get('abilities', {}), 3)}

=== Task ===
Set reasonable hourly wage based on skill match and market rate.

=== Guidelines ===
- Excellent match (>0.8): 110-120% of market rate
- Good match (0.6-0.8): 95-110% of market rate  
- Fair match (0.4-0.6): 85-95% of market rate
- Poor match (<0.4): 75-85% of market rate

=== Response Format ===
{{
    "recommended_wage": wage_value
}}"""
        
        # 计算并打印token数量
        prompt_tokens = calculate_tokens(prompt)
        print(f"💰 [薪资设定] Prompt Token数量: {prompt_tokens}")
        logger.info(f"薪资设定Prompt Token数量: {prompt_tokens}")
        
        try:
            # 初始化LLM
            # llm = LLM()
            
            # # 调用LLM
            # response = await llm.atext_request(prompt)
            
            response = await self.client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度减少幻觉，提高一致性
                stream=False
            )
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM wage setting failed: {e}")
            return {
                "recommended_wage": base_wage,
                "wage_adjustment_factor": 1.0,
                "reasoning": f"LLM call failed, using base wage: {e}",
                "key_strengths": [],
                "key_weaknesses": [],
                "overall_assessment": "Unable to assess"
            }

    async def process_wages(self, economic_center, month: int):
        """
        Processes wages for all jobs in the market.
        This could involve updating household accounts, etc.
        """
        for match in self.matched_jobs:
            await economic_center.process_labor.remote(
                wage_hour=match.average_wage,
                household_id=match.household_id,
                company_id=match.company_id,
                month=month
            ) 
   
    # ===== 新增：多候选人竞争机制 =====
    
    async def submit_job_application(self, job_application: JobApplication, current_month: Optional[int] = None) -> bool:
        """
        提交工作申请
        
        Args:
            job_application: JobApplication对象
            current_month: 当前仿真月份，用于判断是否允许重新申请
            
        Returns:
            bool: 申请是否成功提交
        """
        job_id = job_application.job_id
        
        # 检查工作是否存在且有效
        job = self.get_job_by_id(job_id)
        if not job or not job.is_valid or job.positions_available <= 0:
            logger.warning(f"Job {job_id} not available for application")
            return False
        
        # 初始化该工作的申请列表
        if job_id not in self.job_applications:
            self.job_applications[job_id] = []
        
        # 检查是否在合理时间内重复申请
        # 允许家庭重新申请工作，但需要一定的时间间隔
        
        existing_application = next(
            (app for app in self.job_applications[job_id] 
             if str(app.household_id) == str(job_application.household_id) and app.lh_type == job_application.lh_type and app.month == current_month), 
            None
        )
        
        if existing_application:
            logger.warning(f"Household {job_application.household_id} already applied for job {job_id}")
            return False
        
        # 添加申请
        self.job_applications[job_id].append(job_application)
        # logger.info(f"Job application submitted: household {job_application.household_id} -> job {job_id}")
        
        return True
    
    def get_job_by_id(self, job_id: str) -> Optional[Job]:
        """
        根据job_id获取工作对象
        """
        return next((job for job in self.job_postings if job.job_id == job_id), None)
    
    async def process_job_applications_for_firm(self, company_id: str, current_month: int) -> List[Dict]:
        """
        为特定企业处理所有相关工作的申请
        
        Args:
            company_id: 企业ID
            
        Returns:
            List[Dict]: 企业的招聘决策结果
        """
        firm_decisions = []
        
        # 获取该企业的所有工作
        firm_jobs = [job for job in self.job_postings if job.company_id == company_id and job.is_valid]
        
        for job in firm_jobs:
            if job.job_id in self.job_applications:
                applications = [app for app in self.job_applications[job.job_id] if app.month == current_month]
                if applications:
                    # 企业评估候选人并做决策
                    decision = await self.firm_evaluate_candidates(job, applications)
                    if decision:
                        firm_decisions.append(decision)
        
        return firm_decisions
    
    async def firm_evaluate_candidates(self, job: Job, applications: List[JobApplication]) -> Optional[Dict]:
        """
        企业评估候选人并做出招聘决策
        
        Args:
            job: 工作对象
            applications: 该工作的所有申请
            
        Returns:
            Dict: 招聘决策，包含选中的候选人和最终薪资
        """
        if not applications or job.positions_available <= 0:
            return None
        
        # 为每个候选人计算匹配分数
        candidate_evaluations = []
        
        for application in applications:
            match_score = self.calculate_skill_match_score(
                application.worker_skills,
                application.worker_abilities, 
                job.required_skills,
                job.required_abilities
            )
            
            candidate_evaluations.append({
                "application": application,
                "match_score": match_score,
                "expected_wage": application.expected_wage
            })
        
        # 使用LLM进行最终决策
        decision = await self.llm_firm_hiring_decision(job, candidate_evaluations)
        
        return decision
    
    async def llm_firm_hiring_decision(self, job: Job, candidate_evaluations: List[Dict]) -> Optional[Dict]:
        """
        使用LLM帮助企业做出招聘决策
        
        Args:
            job: 工作对象
            candidate_evaluations: 候选人评估结果
            
        Returns:
            Dict: 招聘决策
        """
        if not candidate_evaluations:
            return None
        
        # 优化的技能要求展示
        def compress_job_requirements(req_dict, max_items=5):
            if not req_dict:
                return "None"
            items = list(req_dict.items())[:max_items]
            return ", ".join([f"{k}({v.get('mean', 50)})" for k, v in items])
        
        # 构建优化的英文prompt
        prompt = f"""=== HR Hiring Decision ===
Position: {job.title} | ${job.wage_per_hour:.2f}/h | {job.positions_available} positions
Required Skills: {compress_job_requirements(job.required_skills)}
Required Abilities: {compress_job_requirements(job.required_abilities)}

=== Candidates ==="""
        
        # 优化候选人数量限制
        max_candidates = 12  # 减少到12个候选人
        if len(candidate_evaluations) > max_candidates:
            # 按技能匹配分数排序，选择最好的候选人
            candidate_evaluations = sorted(candidate_evaluations, 
                                         key=lambda x: x["match_score"], 
                                         reverse=True)[:max_candidates]
            print(f"    ⚠️  Too many candidates, filtered to top {max_candidates} (by match score)")
        
        for i, evaluation in enumerate(candidate_evaluations):
            app = evaluation["application"]
            
            # 优化技能选择 - 重要性5个 + 优势3个，处理交集，最终8个
            def select_display_skills(worker_skills, job_skills, max_items=8):
                """
                优化的技能选择策略（减少token量）：
                1. 从职位要求选择重要性最高的5个技能
                2. 从劳动者选择数值最高的3个技能
                3. 处理交集，补充到8个技能
                """
                if not worker_skills:
                    return []
                if not job_skills:
                    # 没有职位要求时，直接按劳动者技能值排序
                    return sorted(worker_skills.items(), key=lambda x: x[1], reverse=True)[:max_items]
                
                # 第一步：选择职位要求中重要性最高的5个技能
                job_top5_skills = sorted(
                    [(skill, req.get('importance', 1.0)) for skill, req in job_skills.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]  # 重要性最高的5个
                
                # 第二步：选择劳动者数值最高的3个技能
                worker_top3_skills = sorted(
                    worker_skills.items(), 
                    key=lambda x: x[1], reverse=True
                )[:3]  # 数值最高的3个
                
                # 第三步：合并并处理交集
                selected_skills = set()
                result = []
                
                # 添加职位要求的重要技能
                for skill, importance in job_top5_skills:
                    if skill not in selected_skills:
                        result.append((skill, worker_skills[skill]))
                        selected_skills.add(skill)
                
                # 添加劳动者的优势技能（如果不重复）
                for skill, value in worker_top3_skills:
                    if skill not in selected_skills:
                        result.append((skill, value))
                        selected_skills.add(skill)
                
                # 如果还没到8个，补充其他技能
                if len(result) < max_items:
                    remaining_skills = [(k, v) for k, v in worker_skills.items() if k not in selected_skills]
                    remaining_skills.sort(key=lambda x: x[1], reverse=True)  # 按数值排序
                    
                    for skill, value in remaining_skills:
                        if len(result) >= max_items:
                            break
                        result.append((skill, value))
                        selected_skills.add(skill)
                
                return result[:max_items]
            
            def select_display_abilities(worker_abilities, job_abilities, max_items=8):
                """
                优化的能力选择策略（减少token量）：
                1. 从职位要求选择重要性最高的5个能力
                2. 从劳动者选择数值最高的3个能力
                3. 处理交集，补充到8个能力
                """
                if not worker_abilities:
                    return []
                if not job_abilities:
                    # 没有职位要求时，直接按劳动者能力值排序
                    return sorted(worker_abilities.items(), key=lambda x: x[1], reverse=True)[:max_items]
                
                # 第一步：选择职位要求中重要性最高的5个能力
                job_top5_abilities = sorted(
                    [(ability, req.get('importance', 1.0)) for ability, req in job_abilities.items()],
                    key=lambda x: x[1], reverse=True
                )[:5]  # 重要性最高的5个
                
                # 第二步：选择劳动者数值最高的3个能力
                worker_top3_abilities = sorted(
                    worker_abilities.items(), 
                    key=lambda x: x[1], reverse=True
                )[:3]  # 数值最高的3个
                
                # 第三步：合并并处理交集
                selected_abilities = set()
                result = []
                
                # 添加职位要求的重要能力
                for ability, importance in job_top5_abilities:
                    if ability not in selected_abilities:
                        result.append((ability, worker_abilities[ability]))
                        selected_abilities.add(ability)
                
                # 添加劳动者的优势能力（如果不重复）
                for ability, value in worker_top3_abilities:
                    if ability not in selected_abilities:
                        result.append((ability, value))
                        selected_abilities.add(ability)
                
                # 如果还没到8个，补充其他能力
                if len(result) < max_items:
                    remaining_abilities = [(k, v) for k, v in worker_abilities.items() if k not in selected_abilities]
                    remaining_abilities.sort(key=lambda x: x[1], reverse=True)  # 按数值排序
                    
                    for ability, value in remaining_abilities:
                        if len(result) >= max_items:
                            break
                        result.append((ability, value))
                        selected_abilities.add(ability)
                
                return result[:max_items]
            
            # 应用优化选择 - 技能8个，能力8个
            skills_items = select_display_skills(app.worker_skills, job.required_skills, 8)
            abilities_items = select_display_abilities(app.worker_abilities, job.required_abilities, 8)
            
            skills_compact = ", ".join([f"{k}:{v}" for k, v in skills_items]) if skills_items else "None"
            abilities_compact = ", ".join([f"{k}:{v}" for k, v in abilities_items]) if abilities_items else "None"
            
            prompt += f"""
{i+1}. {app.household_id}_{app.lh_type} | ${evaluation["expected_wage"]:.1f}/h | Match:{evaluation["match_score"]:.2f}
   Skills: {skills_compact}
   Abilities: {abilities_compact}"""
        

        
        prompt += f"""

=== Task ===
Select {job.positions_available} primary candidates. Prioritize match score >0.6 and reasonable wages.

=== Response Format (JSON) ===
{{
    "selected_candidates": [
        {{
            "household_id": "exact_id_from_above",
            "lh_type": "head_or_spouse",
            "final_wage": wage_amount,
            "reason": "brief explanation"
        }}
    ],
    "rejected_count": number_of_rejected_candidates
}}

Guidelines: Final wages 0.85-1.15x posted wage.
"""
        
        # 计算并打印token数量
        prompt_tokens = calculate_tokens(prompt)
        # print(f"🏢 [招聘决策] Prompt Token数量: {prompt_tokens} (候选人数量: {len(candidate_evaluations)})")
        
        try:
            response = await self.client.chat.completions.create(
                model=os.getenv("MODEL", ""),
                messages=[
                    {"role": "system", "content": "You are a professional HR manager. Always respond with valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度减少幻觉，提高一致性
                stream=False
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 清理响应内容，提取JSON部分
            if response_content.startswith("```json"):
                start_idx = response_content.find("{")
                end_idx = response_content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_content = response_content[start_idx:end_idx]
            elif response_content.startswith("```"):
                lines = response_content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') or in_json:
                        in_json = True
                        json_lines.append(line)
                        if line.strip().endswith('}') and json_lines:
                            break
                response_content = '\n'.join(json_lines)
            
            result = json.loads(response_content)
            
            # 立即验证LLM响应中的候选人是否有效
            # print(f"    🔍 验证LLM选择的候选人...")
            all_selected_candidates = []
            if "primary_candidates" in result:
                all_selected_candidates.extend(result["primary_candidates"])
            if "backup_candidates" in result:
                all_selected_candidates.extend(result["backup_candidates"])
            
            valid_candidate_keys = set()
            for evaluation in candidate_evaluations:
                app = evaluation["application"]
                key = f"{str(app.household_id)}_{app.lh_type}"
                valid_candidate_keys.add(key)
            
            invalid_selections = []
            for candidate in all_selected_candidates:
                raw_id = str(candidate.get("household_id", "")).strip()
                lh_type = (candidate.get("lh_type") or "head").strip()
                # 规范化：若 household_id 已包含角色后缀，解析并覆盖 lh_type
                norm_id = raw_id
                if "_" in raw_id:
                    try:
                        id_part, role_part = raw_id.rsplit("_", 1)
                        role_part_lower = role_part.lower()
                        if role_part_lower in ("head", "spouse"):
                            norm_id = id_part
                            lh_type = role_part_lower
                    except Exception:
                        pass
                candidate_key = f"{norm_id}_{lh_type}"
                if candidate_key not in valid_candidate_keys:
                    invalid_selections.append(f"{raw_id} -> {candidate_key}")
            
            # if invalid_selections:
            #     print(f"    ⚠️  LLM选择了无效候选人: {', '.join(invalid_selections)}")
            #     print(f"    📋 有效候选人: {list(valid_candidate_keys)}")
            # else:
            #     print(f"    ✅ 所有选择的候选人都有效")
            
            # 验证和处理结果 - 支持新的简化格式和旧格式
            if "selected_candidates" in result:
                # 处理新的简化格式
                selected_candidates = result.get("selected_candidates", [])
                
                # 创建申请查找字典
                application_lookup = {}
                valid_candidate_keys = set()
                for evaluation in candidate_evaluations:
                    app = evaluation["application"]
                    key = f"{str(app.household_id)}_{app.lh_type}"
                    application_lookup[key] = app
                    valid_candidate_keys.add(key)
                
                # 验证选中的候选人
                valid_selected_candidates = []
                for candidate in selected_candidates:
                    raw_id = str(candidate.get("household_id", "")).strip()
                    lh_type = (candidate.get("lh_type") or "head").strip()
                    
                    # 规范化ID
                    norm_id = raw_id
                    if "_" in raw_id:
                        try:
                            id_part, role_part = raw_id.rsplit("_", 1)
                            role_part_lower = role_part.lower()
                            if role_part_lower in ("head", "spouse"):
                                norm_id = id_part
                                lh_type = role_part_lower
                        except Exception:
                            pass
                    
                    candidate_key = f"{norm_id}_{lh_type}"
                    candidate["household_id"] = norm_id
                    candidate["lh_type"] = lh_type
                    
                    if candidate_key in valid_candidate_keys:
                        valid_selected_candidates.append(candidate)
                    else:
                        print(f"    ❌ Invalid candidate selected: {raw_id} -> {candidate_key}")
                
                # 如果没有有效候选人，自动选择最佳候选人
                if not valid_selected_candidates and candidate_evaluations:
                    print(f"    🔄 No valid candidates selected, auto-selecting best candidates...")
                    sorted_candidates = sorted(candidate_evaluations, 
                                             key=lambda x: x["match_score"], 
                                             reverse=True)
                    
                    for i, evaluation in enumerate(sorted_candidates[:job.positions_available]):
                        app = evaluation["application"]
                        auto_candidate = {
                            "household_id": str(app.household_id),
                            "lh_type": app.lh_type,
                            "final_wage": evaluation["expected_wage"],
                            "reason": f"Auto-selected: highest match score ({evaluation['match_score']:.3f})"
                        }
                        valid_selected_candidates.append(auto_candidate)
                
                # 将选中的候选人设置为主要候选人，备选候选人为空
                primary_candidates = valid_selected_candidates
                backup_candidates = []
                all_candidates = primary_candidates
                
            elif "primary_candidates" in result or "backup_candidates" in result:
                # 处理主要候选人
                primary_candidates = result.get("primary_candidates", [])
                backup_candidates = result.get("backup_candidates", [])
                
                # 创建申请查找字典，用于补充lh_type信息和验证候选人
                application_lookup = {}
                valid_candidate_keys = set()
                for evaluation in candidate_evaluations:
                    app = evaluation["application"]
                    key = f"{str(app.household_id)}_{app.lh_type}"  # 确保household_id是字符串
                    application_lookup[key] = app
                    valid_candidate_keys.add(key)
                
                # 过滤和验证候选人，确保只包含实际申请了工作的候选人
                valid_primary_candidates = []
                valid_backup_candidates = []
                
                # 验证主要候选人
                for candidate in primary_candidates:
                    raw_id = str(candidate.get("household_id", "")).strip()  # 兼容 '24' 或 '24_spouse'
                    lh_type = (candidate.get("lh_type") or "head").strip()
                    # 规范化：若 household_id 已包含角色后缀，解析并覆盖 lh_type
                    norm_id = raw_id
                    if "_" in raw_id:
                        try:
                            id_part, role_part = raw_id.rsplit("_", 1)
                            role_part_lower = role_part.lower()
                            if role_part_lower in ("head", "spouse"):
                                norm_id = id_part
                                lh_type = role_part_lower
                        except Exception:
                            pass
                    candidate_key = f"{norm_id}_{lh_type}"
                    # 将规范化后的字段写回，便于后续使用
                    candidate["household_id"] = norm_id
                    candidate["lh_type"] = lh_type

                    if candidate_key in valid_candidate_keys:
                        valid_primary_candidates.append(candidate)
                    # else:
                    #     print(f"    ❌ LLM选择了无效的主要候选人: {raw_id} -> {candidate_key} - 未申请此工作")
                    #     print(f"    📋 有效候选人列表: {list(valid_candidate_keys)}")
                
                # 验证备选候选人
                for candidate in backup_candidates:
                    raw_id = str(candidate.get("household_id", "")).strip()
                    lh_type = (candidate.get("lh_type") or "head").strip()
                    norm_id = raw_id
                    if "_" in raw_id:
                        try:
                            id_part, role_part = raw_id.rsplit("_", 1)
                            role_part_lower = role_part.lower()
                            if role_part_lower in ("head", "spouse"):
                                norm_id = id_part
                                lh_type = role_part_lower
                        except Exception:
                            pass
                    candidate_key = f"{norm_id}_{lh_type}"
                    candidate["household_id"] = norm_id
                    candidate["lh_type"] = lh_type

                    if candidate_key in valid_candidate_keys:
                        valid_backup_candidates.append(candidate)
                    else:
                        print(f"    ❌ LLM选择了无效的备选候选人: {raw_id} -> {candidate_key} - 未申请此工作")
                        print(f"    📋 有效候选人列表: {list(valid_candidate_keys)}")
                
                # 如果主要候选人全部无效，自动选择最佳候选人作为备选方案
                if not valid_primary_candidates and candidate_evaluations:
                    print(f"    🔄 所有主要候选人都无效，自动选择最佳候选人...")
                    # 按技能匹配分数排序，选择最佳候选人
                    sorted_candidates = sorted(candidate_evaluations, 
                                             key=lambda x: x["match_score"], 
                                             reverse=True)
                    
                    for i, evaluation in enumerate(sorted_candidates[:job.positions_available]):
                        app = evaluation["application"]
                        auto_candidate = {
                            "household_id": str(app.household_id),
                            "lh_type": app.lh_type,
                            "final_wage_offer": evaluation["expected_wage"],
                            "selection_reasoning": f"自动选择：技能匹配分数最高 ({evaluation['match_score']:.3f})",
                            "priority_rank": i + 1
                        }
                        valid_primary_candidates.append(auto_candidate)
                        print(f"    ✅ 自动选择候选人: {app.household_id} ({app.lh_type}) - 技能匹配: {evaluation['match_score']:.3f}")
                
                # 如果备选候选人不足，从剩余候选人中补充
                if len(valid_backup_candidates) < 3 and candidate_evaluations:
                    used_candidates = set()
                    for candidate in valid_primary_candidates:
                        used_candidates.add(f"{candidate['household_id']}_{candidate['lh_type']}")
                    for candidate in valid_backup_candidates:
                        used_candidates.add(f"{candidate['household_id']}_{candidate['lh_type']}")
                    
                    # 从剩余候选人中选择最佳的作为备选
                    remaining_candidates = []
                    for evaluation in candidate_evaluations:
                        app = evaluation["application"]
                        candidate_key = f"{str(app.household_id)}_{app.lh_type}"
                        if candidate_key not in used_candidates:
                            remaining_candidates.append(evaluation)
                    
                    # 按技能匹配分数排序
                    remaining_candidates.sort(key=lambda x: x["match_score"], reverse=True)
                    
                    for evaluation in remaining_candidates[:3-len(valid_backup_candidates)]:
                        app = evaluation["application"]
                        auto_backup = {
                            "household_id": str(app.household_id),
                            "lh_type": app.lh_type,
                            "final_wage_offer": evaluation["expected_wage"],
                            "selection_reasoning": f"自动备选：技能匹配分数 {evaluation['match_score']:.3f}",
                            "priority_rank": len(valid_backup_candidates) + 2
                        }
                        valid_backup_candidates.append(auto_backup)
                        print(f"    ✅ 自动选择备选候选人: {app.household_id} ({app.lh_type}) - 技能匹配: {evaluation['match_score']:.3f}")
                
                # 使用验证后的候选人列表
                primary_candidates = valid_primary_candidates
                backup_candidates = valid_backup_candidates
                
                # 确保所有候选人的薪资在合理范围内，并补充lh_type信息
                all_candidates = primary_candidates + backup_candidates
                for candidate in all_candidates:
                    # 补充lh_type信息（如果LLM没有返回）
                    if "lh_type" not in candidate or not candidate.get("lh_type"):
                        household_id = candidate.get("household_id")
                        # 先尝试head，再尝试spouse
                        head_key = f"{household_id}_head"
                        spouse_key = f"{household_id}_spouse"
                        
                        if head_key in application_lookup:
                            candidate["lh_type"] = "head"
                        elif spouse_key in application_lookup:
                            candidate["lh_type"] = "spouse"
                        else:
                            candidate["lh_type"] = "head"  # 默认值
                            logger.warning(f"无法确定候选人 {household_id} 的lh_type，设为默认值 'head'")
                    
                    final_wage = candidate.get("final_wage_offer", job.wage_per_hour)
                    
                    # 数据清理：确保薪资是数字类型
                    if isinstance(final_wage, str):
                        # 移除美元符号和其他非数字字符，只保留数字和小数点
                        final_wage = ''.join(c for c in str(final_wage) if c.isdigit() or c == '.')
                        try:
                            final_wage = float(final_wage) if final_wage else job.wage_per_hour
                        except ValueError:
                            final_wage = job.wage_per_hour
                            logger.warning(f"无法解析薪资字符串，使用默认值: {candidate.get('final_wage_offer', 'unknown')}")
                    
                    min_wage = job.wage_per_hour * 0.8
                    max_wage = job.wage_per_hour * 1.2
                    candidate["final_wage_offer"] = max(min_wage, min(final_wage, max_wage))
                
                # 存储备选候选人供后续使用
                if backup_candidates:
                    if job.job_id not in self.backup_candidates:
                        self.backup_candidates[job.job_id] = []
                    self.backup_candidates[job.job_id].extend(backup_candidates)
                
                return {
                    "job_id": job.job_id,
                    "company_id": job.company_id,
                    "job_title": job.title,
                    "primary_candidates": primary_candidates,
                    "backup_candidates": backup_candidates,
                    "rejection_reasons": result.get("rejection_reasons", {}),
                    "total_candidates": len(candidate_evaluations)
                }
            
            # 兼容旧格式 - 如果LLM返回的是旧格式
            elif "selected_candidates" in result:
                selected = result["selected_candidates"]
                # 将第一个作为主要候选人，其余作为备选
                primary = selected[:1] if selected else []
                backup = selected[1:] if len(selected) > 1 else []
                
                # 创建申请查找字典，用于补充lh_type信息
                application_lookup = {}
                for evaluation in candidate_evaluations:
                    app = evaluation["application"]
                    key = f"{app.household_id}_{app.lh_type}"
                    application_lookup[key] = app
                
                for candidate in selected:
                    # 补充lh_type信息（如果LLM没有返回）
                    if "lh_type" not in candidate or not candidate.get("lh_type"):
                        household_id = candidate.get("household_id")
                        # 先尝试head，再尝试spouse
                        head_key = f"{household_id}_head"
                        spouse_key = f"{household_id}_spouse"
                        
                        if head_key in application_lookup:
                            candidate["lh_type"] = "head"
                        elif spouse_key in application_lookup:
                            candidate["lh_type"] = "spouse"
                        else:
                            candidate["lh_type"] = "head"  # 默认值
                            logger.warning(f"无法确定候选人 {household_id} 的lh_type，设为默认值 'head'")
                    
                    final_wage = candidate.get("final_wage_offer", job.wage_per_hour)
                    
                    # 数据清理：确保薪资是数字类型
                    if isinstance(final_wage, str):
                        # 移除美元符号和其他非数字字符，只保留数字和小数点
                        final_wage = ''.join(c for c in str(final_wage) if c.isdigit() or c == '.')
                        try:
                            final_wage = float(final_wage) if final_wage else job.wage_per_hour
                        except ValueError:
                            final_wage = job.wage_per_hour
                            logger.warning(f"无法解析薪资字符串，使用默认值: {candidate.get('final_wage_offer', 'unknown')}")
                    
                    min_wage = job.wage_per_hour * 0.8
                    max_wage = job.wage_per_hour * 1.2
                    candidate["final_wage_offer"] = max(min_wage, min(final_wage, max_wage))
                
                # 存储备选候选人供后续使用
                if backup:
                    if job.job_id not in self.backup_candidates:
                        self.backup_candidates[job.job_id] = []
                    self.backup_candidates[job.job_id].extend(backup)
                
                return {
                    "job_id": job.job_id,
                    "company_id": job.company_id,
                    "job_title": job.title,
                    "primary_candidates": primary,
                    "backup_candidates": backup,
                    "rejection_reasons": result.get("rejection_reasons", {}),
                    "total_candidates": len(candidate_evaluations)
                }
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM hiring decision JSON parse error for job {job.job_id}: {e}")
            logger.error(f"Raw response: {response_content if 'response_content' in locals() else 'No content'}")
        except Exception as e:
            logger.error(f"LLM hiring decision failed for job {job.job_id}: {e}")
        
        # 如果LLM失败，使用简单的基于分数的选择
        return self.fallback_hiring_decision(job, candidate_evaluations)
    
    def fallback_hiring_decision(self, job: Job, candidate_evaluations: List[Dict]) -> Dict:
        """
        LLM失败时的备选招聘决策逻辑 - 支持备选候选人
        """
        # 按匹配分数排序
        sorted_candidates = sorted(candidate_evaluations, key=lambda x: x["match_score"], reverse=True)
        
        primary_candidates = []
        backup_candidates = []
        positions_to_fill = min(job.positions_available, len(sorted_candidates))
        
        # 选择主要候选人
        for i in range(positions_to_fill):
            if i < len(sorted_candidates):
                candidate = sorted_candidates[i]
                app = candidate["application"]
                
                # 简单的薪资决策：基于匹配分数调整
                match_score = candidate["match_score"]
                if match_score >= 0.8:
                    final_wage = job.wage_per_hour * 1.05  # 5%奖励
                elif match_score >= 0.6:
                    final_wage = job.wage_per_hour
                else:
                    final_wage = job.wage_per_hour * 0.95  # 5%折扣
                
                primary_candidates.append({
                    "household_id": app.household_id,
                    "lh_type": app.lh_type,
                    "final_wage_offer": final_wage,
                    "selection_reasoning": f"Primary selection based on match score {match_score:.3f}",
                    "priority_rank": 1
                })
        
        # 选择备选候选人（接下来的2-3个最佳候选人）
        backup_start = positions_to_fill
        backup_count = min(3, len(sorted_candidates) - backup_start)  # 最多3个备选
        
        for i in range(backup_start, backup_start + backup_count):
            candidate = sorted_candidates[i]
            app = candidate["application"]
            
            match_score = candidate["match_score"]
            if match_score >= 0.7:
                final_wage = job.wage_per_hour
            elif match_score >= 0.5:
                final_wage = job.wage_per_hour * 0.95
            else:
                final_wage = job.wage_per_hour * 0.90
            
            backup_candidates.append({
                "household_id": app.household_id,
                "lh_type": app.lh_type,
                "final_wage_offer": final_wage,
                "selection_reasoning": f"Backup selection based on match score {match_score:.3f}",
                "priority_rank": i - backup_start + 2
            })
        
        return {
            "job_id": job.job_id,
            "company_id": job.company_id,
            "job_title": job.title,
            "primary_candidates": primary_candidates,
            "backup_candidates": backup_candidates,
            "rejection_reasons": {},
            "total_candidates": len(candidate_evaluations)
        }
    
    async def finalize_hiring_decisions(self, hiring_decisions: List[Dict]) -> List[Dict]:
        """
        确认招聘决策并更新工作状态
        
        注意：这个方法现在只是简单确认企业的招聘决策，不处理重复招聘问题。
        重复招聘的处理应该在后续的家庭接受/拒绝机制中处理。
        
        Args:
            hiring_decisions: 招聘决策列表
            
        Returns:
            List[Dict]: 企业发出的所有job offers
        """
        job_offers = []  # 改名为job_offers，表示这些是企业发出的offer
        
        for decision in hiring_decisions:
            job_id = decision["job_id"]
            job = self.get_job_by_id(job_id)
            
            if not job:
                continue
            
            # 只给主要候选人发送初始offer，备选候选人暂时保存
            primary_candidates = decision.get("primary_candidates", [])
            backup_candidates = decision.get("backup_candidates", [])
            
            # 兼容旧格式
            if not primary_candidates and "selected_candidates" in decision:
                primary_candidates = decision["selected_candidates"][:1]  # 第一个作为主要
                backup_candidates = decision["selected_candidates"][1:]   # 其余作为备选
            
            # 给主要候选人发送offer
            for candidate in primary_candidates:
                job_offers.append({
                    "job_id": job_id,
                    "household_id": candidate["household_id"],
                    "lh_type": candidate["lh_type"],
                    "offered_wage": candidate["final_wage_offer"],
                    "job_title": job.title,
                    "company_id": job.company_id,
                    "job_description": job.description,
                    "hours_per_period": job.hours_per_period,
                    "offer_status": "pending"  # pending, accepted, rejected
                })
                
                print(f"    📧 企业 {job.company_id} 向家庭 {candidate['household_id']} ({candidate['lh_type']}) 发出offer:")
                print(f"        职位: {job.title}")
                print(f"        薪资: ${candidate['final_wage_offer']:.2f}/小时")
            
            # 保存备选候选人信息，以备主要候选人拒绝时使用
            if backup_candidates:
                if job_id not in self.backup_candidates:
                    self.backup_candidates[job_id] = []
                # 清空之前的备选候选人，使用当前决策的结果
                self.backup_candidates[job_id] = []
                for backup in backup_candidates:
                    # 验证这个备选候选人确实申请了这个工作
                    backup_household_id = str(backup["household_id"])  # 确保是字符串
                    backup_lh_type = backup["lh_type"]
                    
                    # 检查是否存在对应的申请记录
                    valid_backup = False
                    if job_id in self.job_applications:
                        for app in self.job_applications[job_id]:
                            if (str(app.household_id) == backup_household_id and 
                                app.lh_type == backup_lh_type):
                                valid_backup = True
                                break
                    
                    # 额外验证：检查这个家庭ID是否在系统中存在
                    if valid_backup:
                        # 这里可以添加额外的家庭存在性检查
                        # 例如检查household_id是否在有效的家庭列表中
                        pass
                    
                    if valid_backup:
                        backup_info = {
                            "household_id": backup["household_id"],
                            "lh_type": backup["lh_type"],
                            "offered_wage": backup["final_wage_offer"],
                            "priority_rank": backup.get("priority_rank", 2),
                            "selection_reasoning": backup.get("selection_reasoning", ""),
                            "job_title": job.title,
                            "company_id": job.company_id,
                            "job_description": job.description,
                            "hours_per_period": job.hours_per_period
                        }
                        self.backup_candidates[job_id].append(backup_info)
                    else:
                        # 检查这个家庭ID是否在任何申请记录中存在
                        household_exists = False
                        for other_job_id, apps in self.job_applications.items():
                            for app in apps:
                                if str(app.household_id) == backup_household_id:
                                    household_exists = True
                                    break
                            if household_exists:
                                break
                        
                        if household_exists:
                            print(f"    ⚠️  跳过无效备选候选人: 家庭 {backup_household_id} ({backup_lh_type}) 没有申请工作 {job.title}")
                        else:
                            print(f"    ❌ 跳过无效备选候选人: 家庭 {backup_household_id} ({backup_lh_type}) 不存在于系统中")
                
                valid_backups_count = len(self.backup_candidates[job_id])
                print(f"    🔄 为工作 {job.title} 保存了 {valid_backups_count} 个有效备选候选人 (原始: {len(backup_candidates)} 个)")
        
        print(f"\n📬 共发出 {len(job_offers)} 个工作offer")
        print(f"💾 共保存 {sum(len(backups) for backups in self.backup_candidates.values())} 个备选候选人")
        
        return job_offers
    
    def _is_backup_candidate_available(self, backup_candidate: Dict, accepted_offers: List[Dict]) -> bool:
        """
        检查备选候选人是否可用（即是否已经接受了其他工作）
        
        Args:
            backup_candidate: 备选候选人信息
            accepted_offers: 所有已接受的工作offers
            
        Returns:
            bool: True表示可用，False表示已被占用
        """
        candidate_key = f"{backup_candidate['household_id']}_{backup_candidate.get('lh_type', 'head')}"
        
        # 检查该候选人是否已经接受了其他工作
        for accepted_offer in accepted_offers:
            accepted_key = f"{accepted_offer['household_id']}_{accepted_offer.get('lh_type', 'head')}"
            if candidate_key == accepted_key:
                return False  # 该候选人已经接受了其他工作
        
        return True  # 候选人可用

    async def process_rejected_offers_and_activate_backups(self, all_offers: List[Dict], accepted_offers: List[Dict]) -> List[Dict]:
        """
        处理被拒绝的offers，激活备选候选人
        
        Args:
            all_offers: 所有发出的offers
            accepted_offers: 家庭接受的offers
            
        Returns:
            List[Dict]: 给备选候选人发出的新offers
        """
        # 找出被拒绝的offers
        accepted_offer_keys = {f"{offer['job_id']}_{offer['household_id']}_{offer['lh_type']}" for offer in accepted_offers}
        rejected_offers = []
        
        for offer in all_offers:
            offer_key = f"{offer['job_id']}_{offer['household_id']}_{offer['lh_type']}"
            if offer_key not in accepted_offer_keys:
                rejected_offers.append(offer)
        
        if not rejected_offers:
            print("✅ 没有被拒绝的offers，无需启用备选候选人")
            return []
        
        print(f"\n🔄 处理 {len(rejected_offers)} 个被拒绝的offers...")
        new_backup_offers = []
        
        for rejected_offer in rejected_offers:
            job_id = rejected_offer["job_id"]
            
            # 检查是否有备选候选人
            if job_id in self.backup_candidates and self.backup_candidates[job_id]:
                # 按优先级排序备选候选人
                backup_list = sorted(self.backup_candidates[job_id], key=lambda x: x.get("priority_rank", 999))
                
                # 过滤出可用的备选候选人（未接受其他工作的）
                available_backups = []
                for backup in backup_list:
                    if self._is_backup_candidate_available(backup, accepted_offers):
                        available_backups.append(backup)
                    else:
                        print(f"        ⚠️  备选候选人 {backup['household_id']} ({backup.get('lh_type', 'head')}) 已接受其他工作，跳过")
                
                if available_backups:
                    # 选择优先级最高的可用备选候选人
                    best_backup = available_backups[0]
                    
                    # 创建新的offer
                    new_offer = {
                        "job_id": job_id,
                        "household_id": best_backup["household_id"],
                        "lh_type": best_backup.get("lh_type", rejected_offer.get("lh_type", "head")),
                        "offered_wage": best_backup["offered_wage"],
                        "job_title": best_backup["job_title"],
                        "company_id": best_backup["company_id"],
                        "job_description": best_backup["job_description"],
                        "hours_per_period": best_backup["hours_per_period"],
                        "offer_status": "backup_activated",
                        "original_candidate": f"{rejected_offer['household_id']} ({rejected_offer['lh_type']})",
                        "backup_reason": "Primary candidate rejected the offer"
                    }
                    
                    new_backup_offers.append(new_offer)
                    
                    # 从备选列表中移除已使用的候选人
                    self.backup_candidates[job_id].remove(best_backup)
                    
                    print(f"    🔄 工作 '{best_backup['job_title']}' 启用备选候选人:")
                    print(f"        原候选人: {rejected_offer['household_id']} ({rejected_offer['lh_type']}) [已拒绝]")
                    print(f"        备选候选人: {best_backup['household_id']} ({best_backup['lh_type']}) [已激活]")
                    print(f"        薪资: ${best_backup['offered_wage']:.2f}/小时")
                else:
                    if backup_list:
                        print(f"    ❌ 工作 '{rejected_offer['job_title']}' 的所有备选候选人都已接受其他工作")
                    else:
                        print(f"    ❌ 工作 '{rejected_offer['job_title']}' 没有可用的备选候选人")
            else:
                print(f"    ❌ 工作 '{rejected_offer['job_title']}' 没有备选候选人")
        
        if new_backup_offers:
            print(f"\n🎯 成功激活 {len(new_backup_offers)} 个备选候选人")
        else:
            print(f"\n⚠️  无法为任何被拒绝的工作找到备选候选人")
        
        return new_backup_offers
    
    async def process_job_acceptances(self, accepted_offers: List[Dict]) -> List[Dict]:
        """
        处理家庭接受的工作offer，完成最终的雇佣确认
        这个方法会在家庭做出接受/拒绝决策后调用
        
        Args:
            accepted_offers: 家庭接受的job offers列表
            
        Returns:
            List[Dict]: 最终确认的雇佣关系
        """
        confirmed_hires = []
        hired_households = set()  # 跟踪已被雇佣的家庭成员
        
        # 按某种优先级排序（比如薪资高的优先）
        sorted_offers = sorted(accepted_offers, key=lambda x: x.get("offered_wage", 0), reverse=True)
        
        for offer in sorted_offers:
            household_key = f"{offer['household_id']}_{offer['lh_type']}"
            job = self.get_job_by_id(offer["job_id"])
            
            if not job or job.positions_available <= 0:
                print(f"    ❌ 职位 '{offer['job_title']}' 已无可用位置")
                continue
            
            # 检查这个家庭成员是否已经被雇佣
            if household_key in hired_households:
                print(f"    ⚠️  家庭 {offer['household_id']} ({offer['lh_type']}) 已被雇佣，跳过重复招聘")
                continue
            
            # 确认雇佣
            job.positions_available -= 1
            
            # 记录已雇佣的家庭成员
            hired_households.add(household_key)
            
            # 记录匹配结果
            matched_job = MatchedJob.create(
                job=job,
                average_wage=offer["offered_wage"],
                household_id=offer["household_id"],
                lh_type=offer["lh_type"],
                company_id=job.company_id
            )
            self.matched_jobs.append(matched_job)
            
            confirmed_hires.append({
                "job_id": offer["job_id"],
                "household_id": offer["household_id"],
                "lh_type": offer["lh_type"],
                "final_wage": offer["offered_wage"],
                "job_title": job.title,
                "company_id": job.company_id,
                "job_SOC": job.SOC,
                "offer_status": offer.get("offer_status", "pending")  # 保留offer状态信息
            })
            
            logger.info(f"Final hiring confirmed: {offer['household_id']} -> {job.title} at ${offer['offered_wage']:.2f}/hour")
        
        skipped_count = len(accepted_offers) - len(confirmed_hires)
        if skipped_count > 0:
            print(f"\n✅ 最终雇佣确认完成，跳过了 {skipped_count} 个重复/无效的接受")
        
        return confirmed_hires
    
    def process_dismissals(self, dismissed_workers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理辞退工人，更新MatchedJob和Job状态
        
        Args:
            dismissed_workers: 被辞退的工人信息列表
            
        Returns:
            Dict: 处理结果统计
        """
        if not dismissed_workers:
            return {'dismissed_count': 0, 'jobs_reopened': 0, 'matched_jobs_removed': 0}
        
        dismissed_count = len(dismissed_workers)
        jobs_reopened = 0
        matched_jobs_removed = 0
        
        print(f"🔄 处理 {dismissed_count} 个被辞退工人的工作状态...")
        
        # 创建辞退工人的查找字典
        dismissed_lookup = {}
        for worker in dismissed_workers:
            key = f"{worker['household_id']}_{worker['lh_type']}"
            dismissed_lookup[key] = worker
        
        # 1. 从matched_jobs中移除被辞退的工人
        original_matched_count = len(self.matched_jobs)
        self.matched_jobs = [
            mj for mj in self.matched_jobs 
            if f"{mj.household_id}_{mj.lh_type}" not in dismissed_lookup
        ]
        matched_jobs_removed = original_matched_count - len(self.matched_jobs)
        
        # 2. 更新相应Job的positions_available和is_valid状态
        job_updates = {}
        for worker in dismissed_workers:
            job_soc = worker['job_SOC']
            company_id = worker['company_id']
            
            # 找到对应的Job并更新可用位置
            for job in self.job_postings:
                if job.SOC == job_soc and job.company_id == company_id:
                    job.positions_available += 1  # 增加可用位置
                    if not job.is_valid:  # 如果之前因为满员而无效，现在重新激活
                        job.is_valid = True
                        jobs_reopened += 1
                    
                    # 记录更新信息
                    job_key = f"{company_id}_{job_soc}"
                    if job_key not in job_updates:
                        job_updates[job_key] = {
                            'job_title': job.title,
                            'company_id': company_id,
                            'positions_freed': 0,
                            'now_available': job.positions_available
                        }
                    job_updates[job_key]['positions_freed'] += 1
                    break
        
        # 打印详细信息
        print(f"   📊 移除了 {matched_jobs_removed} 个MatchedJob记录")
        print(f"   📊 重新开放了 {jobs_reopened} 个工作岗位")
        
        if job_updates:
            print(f"   📋 工作岗位更新详情:")
            for job_key, info in job_updates.items():
                print(f"      {info['job_title']} ({info['company_id']}): "
                      f"释放 {info['positions_freed']} 个位置, "
                      f"现有 {info['now_available']} 个可用位置")
        
        return {
            'dismissed_count': dismissed_count,
            'jobs_reopened': jobs_reopened,
            'matched_jobs_removed': matched_jobs_removed,
            'job_updates': job_updates
        }
    
    def get_total_job_positions(self) -> Dict[str, int]:
        """
        获取总岗位数统计
        
        Returns:
            Dict: 岗位统计信息
        """
        total_positions = 0
        available_positions = 0
        filled_positions = 0
        
        for job in self.job_postings:
            total_positions += job.positions_available
            if job.is_valid:
                available_positions += job.positions_available
            
        filled_positions = len(self.matched_jobs)
        
        return {
            'total_positions': total_positions,
            'available_positions': available_positions,
            'filled_positions': filled_positions,
            'vacancy_rate': available_positions / total_positions if total_positions > 0 else 0.0
        }
    
    def get_employment_statistics(self, households: List = None) -> Dict[str, int]:
        """
        获取就业统计信息 - 基于MatchedJob和家庭labor_hour状态
        
        Args:
            households: 家庭对象列表（可选，用于获取完整的劳动力统计）
            
        Returns:
            Dict: 就业统计信息
        """
        # 基于MatchedJob的就业统计
        employed_count = len(self.matched_jobs)
        
        # 如果有家庭对象列表，可以获取完整统计
        total_labor_force = 0
        if households:
            for household in households:
                total_labor_force += len(household.labor_hours)
        else:
            # 否则基于job postings估算
            total_labor_force = sum(job.positions_available for job in self.job_postings)
            total_labor_force += employed_count  # 加上已就业的
        
        unemployed_count = total_labor_force - employed_count
        
        return {
            'employed': employed_count,
            'unemployed': unemployed_count,
            'total_labor_force': total_labor_force,
            'unemployment_rate': unemployed_count / total_labor_force if total_labor_force > 0 else 0.0
        }
    
    async def dismiss_workers_randomly(self, dismissal_rate: float = 0.1, month: int = 1) -> Dict[str, Any]:
        """
        随机辞退工人 - 正确的架构版本
        
        Args:
            dismissal_rate: 辞退比例 (默认10%)
            month: 当前月份
            households: 家庭对象列表，用于通知更新labor_hour
            firms: 企业对象列表，用于更新员工数量
            
        Returns:
            Dict: 辞退结果统计
        """
        import random
        if not self.matched_jobs:
            print(f"📊 第 {month} 月无匹配工作，跳过辞退")
            return {'dismissed_count': 0, 'jobs_reopened': 0}
        
        # 计算要辞退的工人数量
        total_employed = len(self.matched_jobs)
        dismiss_count = int(total_employed * dismissal_rate)
        
        if dismiss_count == 0:
            print(f"📊 第 {month} 月辞退数量为0，跳过辞退")
            return {'dismissed_count': 0, 'jobs_reopened': 0}
        
        # 随机选择要辞退的MatchedJob
        matched_jobs_to_dismiss = random.sample(self.matched_jobs, dismiss_count)
        
        print(f"🔥 第 {month} 月开始辞退 {dismiss_count}/{total_employed} 个工人 (辞退率: {dismissal_rate:.1%})")
        
        dismissed_workers = []
        jobs_reopened = 0
        firm_updates = {}
        
        for matched_job in matched_jobs_to_dismiss:
            try:
                household_id = matched_job.household_id
                lh_type = matched_job.lh_type
                company_id = matched_job.company_id
                job_soc = matched_job.job.SOC
                job_title = matched_job.job.title
                
                
                for job_posting in self.job_postings:
                    if job_posting.SOC == job_soc and job_posting.company_id == company_id:
                        job_posting.positions_available += 1
                        jobs_reopened += 1  # 更新重新开放岗位计数
                        print(f"   🔄 岗位: {job_posting.title} ({job_posting.company_id}) 增加一个位置")
                        break

                
                # 3. 记录需要更新的企业员工数量（不在这里直接修改）
                if company_id not in firm_updates:
                    firm_updates[company_id] = {'count': 0, 'firm_name': company_id}
                firm_updates[company_id]['count'] += 1
                
                dismissed_info = {
                    'household_id': household_id,
                    'lh_type': lh_type,
                    'company_id': company_id,
                    'job_title': job_title,
                    'job_SOC': job_soc,
                    'month': month
                }
                
                dismissed_workers.append(dismissed_info)
                
            except Exception as e:
                print(f"❌ 辞退MatchedJob失败: {e}")
                continue
        
        # 4. 从matched_jobs中移除被辞退的工人
        dismissed_keys = {f"{mj.household_id}_{mj.lh_type}" for mj in matched_jobs_to_dismiss}
        self.matched_jobs = [
            mj for mj in self.matched_jobs 
            if f"{mj.household_id}_{mj.lh_type}" not in dismissed_keys
        ]
        
        # 打印统计信息
        actual_dismissed = len(dismissed_workers)
        print(f"✅ 辞退完成，实际辞退 {actual_dismissed} 人")
        print(f"📊 重新开放了 {jobs_reopened} 个工作岗位")
        
        return {
            'dismissed_count': actual_dismissed,
            'jobs_reopened': jobs_reopened,
            'firm_updates': firm_updates,
            'dismissed_workers': dismissed_workers,
        }
    
    async def dismiss_workers_by_firm(self, firms_to_dismiss: List[Dict], month: int = 1) -> Dict[str, Any]:
        """
        基于企业利润的智能辞退
        
        Args:
            firms_to_dismiss: 要辞退的企业列表，每个元素包含 {'company_id', 'firm', 'profit', 'employees'}
            month: 当前月份
            
        Returns:
            Dict: 辞退结果统计
        """
        if not self.matched_jobs:
            print(f"📊 第 {month} 月无匹配工作，跳过辞退")
            return {'dismissed_count': 0, 'jobs_reopened': 0, 'firm_updates': {}, 'dismissed_workers': []}
        
        dismissed_workers = []
        jobs_reopened = 0
        firm_updates = {}
        
        print(f"🔥 第 {month} 月开始基于企业利润的智能辞退")
        
        for firm_data in firms_to_dismiss:
            firm_id = firm_data['company_id']  # 修复：使用 'company_id' 而非 'firm_id'
            firm = firm_data['firm']
            profit = firm_data['profit']
            employees = firm_data['employees']
            
            print(f"   📊 处理企业 {firm_id}: 利润${profit:.2f}, 员工{employees}人")
            
            # 找到该企业的所有匹配工作
            firm_matched_jobs = [mj for mj in self.matched_jobs if mj.company_id == firm_id]
            
            if not firm_matched_jobs:
                print(f"   ⚠️  企业 {firm_id} 没有匹配的员工，跳过")
                continue
            
            # 随机选择1个员工进行辞退
            import random
            if len(firm_matched_jobs) > 0:
                matched_job_to_dismiss = random.choice(firm_matched_jobs)
                
                try:
                    household_id = matched_job_to_dismiss.household_id
                    lh_type = matched_job_to_dismiss.lh_type
                    company_id = matched_job_to_dismiss.company_id
                    job_soc = matched_job_to_dismiss.job.SOC
                    job_title = matched_job_to_dismiss.job.title
                    
                    # 记录辞退信息
                    dismissed_workers.append({
                        'household_id': household_id,
                        'lh_type': lh_type,
                        'company_id': company_id,
                        'job_SOC': job_soc,
                        'job_title': job_title
                    })
                    
                    # 重新开放岗位
                    for job_posting in self.job_postings:
                        if job_posting.SOC == job_soc and job_posting.company_id == company_id:
                            job_posting.positions_available += 1
                            jobs_reopened += 1
                            print(f"   🔄 岗位: {job_posting.title} ({job_posting.company_id}) 增加一个位置")
                            break
                    
                    # 记录企业更新
                    if company_id not in firm_updates:
                        firm_updates[company_id] = {'count': 0}
                    firm_updates[company_id]['count'] += 1
                    
                    # 从matched_jobs中移除
                    self.matched_jobs = [mj for mj in self.matched_jobs if mj != matched_job_to_dismiss]
                    
                    print(f"   ✅ 企业 {firm_id} 辞退1名员工: {household_id} ({lh_type})")
                    
                except Exception as e:
                    print(f"   ❌ 辞退企业 {firm_id} 员工失败: {e}")
                    continue
        
        # 打印统计信息
        actual_dismissed = len(dismissed_workers)
        print(f"✅ 智能辞退完成，实际辞退 {actual_dismissed} 人")
        print(f"📊 重新开放了 {jobs_reopened} 个工作岗位")
        
        return {
            'dismissed_count': actual_dismissed,
            'jobs_reopened': jobs_reopened,
            'firm_updates': firm_updates,
            'dismissed_workers': dismissed_workers,
        }
    
