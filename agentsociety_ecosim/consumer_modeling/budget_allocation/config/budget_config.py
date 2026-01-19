"""
预算分配配置模块

本模块包含预算分配系统的所有配置常量：
- 预算类别定义
- 类别名称映射
- 属性到类别的映射
- 预算到商品分类的映射
- 历史数据转换配置

作者：Agent Society Ecosim Team
日期：2025-10-22
"""

from typing import Dict, List

from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

class BudgetCategories:
    """预算类别定义配置类"""
    
    # ============================================================================
    # 预算类别定义（共17个）
    # ============================================================================
    # 说明：将healthcare, transportation, education拆分为商品和服务两部分
    # 商品部分：可在沃尔玛等零售商购买的实物商品
    # 服务部分：医疗服务、汽油保险、学费等无形服务
    # 拆分依据：基于BLS Consumer Expenditure Survey 2022统计数据
    # ============================================================================
    
    CATEGORY_KEYS = [
        # ========== 商品类支出（9个）- 可在零售商购买 ==========
        "food_expenditure",                    # 食品与日用品
        "clothing_expenditure",                # 服装与美妆
        "childcare_expenditure",               # 儿童保育用品
        "electronics_expenditure",             # 电子产品
        "home_furnishing_equipment",           # 家居设备
        "other_recreation_expenditure",        # 其他娱乐用品
        
        # 拆分后的商品类别（从混合类别分离出来）
        "healthcare_goods_expenditure",        # 医疗保健商品（OTC药品、保健品、医疗用品）
        "transportation_goods_expenditure",    # 交通工具商品（汽车配件、自行车等）
        "education_goods_expenditure",         # 教育用品商品（教材、文具、书包）
        
        # ========== 服务类支出（8个）- 无对应实物商品 ==========
        "housing_expenditure",                 # 住房支出（租金/房贷）
        "utilities_expenditure",               # 公用事业（水电煤气）
        
        # 拆分后的服务类别（从混合类别分离出来）
        "healthcare_services_expenditure",     # 医疗服务（医保、门诊、住院、处方药）
        "transportation_services_expenditure", # 交通服务（汽油、保险、维修保养、停车费）
        "education_services_expenditure",      # 教育服务（学费、培训费、在线课程、考试费）
        
        "travel_expenditure",                  # 旅行支出
        "phone_internet_expenditure",          # 通讯网络
    ]
    
    # ============================================================================
    # 旧版预算类别定义（13个）- 用于处理历史PSID数据
    # ============================================================================
    LEGACY_CATEGORY_KEYS = [
        "food_expenditure",
        "clothing_expenditure",
        "education_expenditure",           # 混合类别（包含商品+服务）
        "childcare_expenditure",
        "electronics_expenditure",
        "home_furnishing_equipment",
        "other_recreation_expenditure",
        "housing_expenditure",
        "utilities_expenditure",
        "transportation_expenditure",      # 混合类别（包含商品+服务）
        "healthcare_expenditure",          # 混合类别（包含商品+服务）
        "travel_expenditure",
        "phone_internet_expenditure",
    ]
    
    # ============================================================================
    # 预算类别中文名称映射
    # ============================================================================
    CATEGORY_NAMES_ZH = {
        # 商品类别
        "food_expenditure": "食品与日用品",
        "clothing_expenditure": "服装与美妆",
        "childcare_expenditure": "儿童保育用品",
        "electronics_expenditure": "电子产品",
        "home_furnishing_equipment": "家居设备",
        "other_recreation_expenditure": "其他娱乐用品",
        
        # 拆分后的商品类别
        "healthcare_goods_expenditure": "医疗保健商品",
        "transportation_goods_expenditure": "交通工具商品",
        "education_goods_expenditure": "教育用品商品",
        
        # 服务类别
        "housing_expenditure": "住房支出",
        "utilities_expenditure": "公用事业",
        "healthcare_services_expenditure": "医疗服务",
        "transportation_services_expenditure": "交通服务",
        "education_services_expenditure": "教育服务",
        "travel_expenditure": "旅行支出",
        "phone_internet_expenditure": "通讯网络",
    }


class AttributeToCategoryMapping:
    """属性到消费类别的映射配置类"""
    
    # ============================================================================
    # 属性到消费类别的映射（用于属性引导预算分配）
    # ============================================================================
    # 说明：定义各种需求属性与预算类别的对应关系
    # importance等级: critical(关键) > high(高) > medium(中) > low(低)
    # ============================================================================
    ATTRIBUTE_TO_CATEGORY_MAPPING = {
        "hunger_satisfaction": {
            "primary": ["food_expenditure"],
            "importance": "critical"
        },
        "thirst_quenching": {
            "primary": ["food_expenditure"],
            "importance": "critical"
        },
        "shelter_protection": {
            "primary": ["housing_expenditure", "home_furnishing_equipment"],
            "importance": "critical"
        },
        "health_maintenance": {
            # 健康维护：主要是医疗商品和服务
            "primary": ["healthcare_goods_expenditure", "healthcare_services_expenditure"],
            "secondary": ["food_expenditure"],
            "importance": "high"
        },
        "nutrition_energy": {
            "primary": ["food_expenditure"],
            "importance": "high"
        },
        "nutrition_health": {
            # 营养健康：食品+医疗保健商品（如保健品）
            "primary": ["food_expenditure", "healthcare_goods_expenditure"],
            "importance": "high"
        },
        "comfort_convenience": {
            # 舒适便利：家居、服装+交通工具商品
            "primary": ["home_furnishing_equipment", "clothing_expenditure"],
            "secondary": ["electronics_expenditure", "transportation_goods_expenditure"],
            "importance": "medium"
        },
        "entertainment_recreation": {
            "primary": ["other_recreation_expenditure", "travel_expenditure"],
            "importance": "medium"
        },
        "social_connection": {
            "primary": ["other_recreation_expenditure", "phone_internet_expenditure"],
            "secondary": ["travel_expenditure"],
            "importance": "medium"
        },
        "learning_growth": {
            # 学习成长：教育商品（教材文具）和教育服务（学费培训）
            "primary": ["education_goods_expenditure", "education_services_expenditure"],
            "secondary": ["other_recreation_expenditure"],
            "importance": "medium"
        },
        "status_recognition": {
            # 地位认同：服装、电子产品+交通工具商品
            "primary": ["clothing_expenditure", "electronics_expenditure"],
            "secondary": ["transportation_goods_expenditure"],
            "importance": "low"
        },
        "achievement_pride": {
            # 成就自豪：娱乐、教育
            "primary": ["other_recreation_expenditure"],
            "secondary": ["education_goods_expenditure", "education_services_expenditure", "electronics_expenditure"],
            "importance": "low"
        },
        "aesthetic_pleasure": {
            "primary": ["home_furnishing_equipment", "clothing_expenditure"],
            "secondary": ["other_recreation_expenditure"],
            "importance": "low"
        }
    }


class CategoryToWalmartMapping:
    """预算类别到沃尔玛商品分类的映射配置类"""
    
    # ============================================================================
    # 预算类别到沃尔玛商品分类的映射
    # ============================================================================
    # 说明：
    # 1. 只映射有实物商品的类别（9个商品类）
    # 2. 服务类别（8个）无商品映射，用空列表表示
    # 3. 映射覆盖沃尔玛所有34个level1商品分类
    # 4. 商品数量基于 products.csv（共29,120件商品）
    # ============================================================================
    BUDGET_TO_WALMART_MAIN = {
        # ========== 纯商品类别 ==========
        "food_expenditure": [
            "food",                  # 3,808件 - 食品
            "household essentials",  # 2,427件 - 家庭日用品
            "personal care"          # 2,265件 - 个人护理用品（牙膏、洗发水等）
        ],
        
        "clothing_expenditure": [
            "clothing",              # 386件 - 服装
            "jewelry",               # 4件 - 珠宝首饰
            "beauty",                # 818件 - 美妆护肤
            "premium beauty"         # 364件 - 高端美妆
        ],
        
        "childcare_expenditure": [
            "baby",                  # 2,303件 - 婴儿用品
            "toys",                  # 828件 - 玩具
            "character shop",        # 3件 - 角色周边
            "shop by movie",         # 3件 - 电影周边
            "shop by video game"     # 1件 - 游戏周边
        ],
        
        "electronics_expenditure": [
            "electronics",           # 49件 - 电子产品
            "cell phones",           # 2件 - 手机
            "video games"            # 7件 - 电子游戏
        ],
        
        "home_furnishing_equipment": [
            "home",                  # 669件 - 家居用品
            "home improvement",      # 114件 - 家装建材
            "patio & garden"         # 110件 - 庭院园艺
        ],
        
        "other_recreation_expenditure": [
            "sports & outdoors",     # 9,672件 - 运动户外
            "pets",                  # 1,284件 - 宠物用品
            "party & occasions",     # 115件 - 派对用品
            "arts crafts & sewing",  # 8件 - 手工艺品（注意：无逗号）
            "music",                 # 2件 - 音乐
            "collectibles",          # 1件 - 收藏品
            "musical instruments",   # 1件 - 乐器
            "shop by brand",         # 137件 - 品牌专区
            "industrial & scientific", # 97件 - 工业科学
            "seasonal",              # 56件 - 季节性商品
            "feature",               # 14件 - 特色商品
            "walmart for business"   # 2件 - 商业采购
        ],
        
        # ========== 拆分后的商品类别 ==========
        "healthcare_goods_expenditure": [
            "health"                 # 3,413件 - 医疗保健商品
            # 包含：OTC药品、保健品、维生素、医疗器械、
            #      医疗用品、家庭护理用品等
        ],
        
        "transportation_goods_expenditure": [
            "auto & tires"           # 133件 - 交通工具商品
            # 包含：汽车配件、电池、摩托车用品、
            #      自行车及配件、车载用品等
        ],
        
        "education_goods_expenditure": [
            "office supplies",       # 9件 - 办公文具
            "books"                  # 15件 - 图书教材
            # 包含：教材、参考书、文具、书包、学习用品等
        ],
        
        # ========== 服务类别（无商品映射）==========
        "housing_expenditure": [],
        "utilities_expenditure": [],
        "healthcare_services_expenditure": [],
        "transportation_services_expenditure": [],
        "education_services_expenditure": [],
        "travel_expenditure": [],
        "phone_internet_expenditure": []
    }
    
    # ============================================================================
    # 无二级分类的支出类别（服务类）
    # ============================================================================
    # 说明：这些类别只分配大类预算，不再往下细分到具体商品
    # 原因：这些是服务类支出，没有对应的实物商品可购买
    # ============================================================================
    NO_SUBCAT_CATEGORIES = {
        # 纯服务类别
        "housing_expenditure": "住房支出",
        "utilities_expenditure": "公用事业",
        "travel_expenditure": "旅行支出",
        "phone_internet_expenditure": "通讯网络",
        
        # 拆分后的服务类别
        "healthcare_services_expenditure": "医疗服务",
        "transportation_services_expenditure": "交通服务",
        "education_services_expenditure": "教育服务"
    }
    
    @classmethod
    def get_budget_to_walmart_main(cls) -> Dict[str, List[str]]:
        """
        获取唯一化处理后的预算类别到沃尔玛商品分类的映射
        
        说明：确保每个沃尔玛商品分类只属于一个预算类别
        原则：如果有重复，优先归属到前面定义的类别
        """
        _subcat_to_main = {}
        _unique_budget_to_walmart_main = {k: [] for k in cls.BUDGET_TO_WALMART_MAIN}
        for main, subcats in cls.BUDGET_TO_WALMART_MAIN.items():
            for subcat in subcats:
                if subcat not in _subcat_to_main:
                    _subcat_to_main[subcat] = main
                    _unique_budget_to_walmart_main[main].append(subcat)
        return _unique_budget_to_walmart_main
    
    @classmethod
    def get_main_to_budget_mapping(cls) -> Dict[str, str]:
        """
        创建反向映射：沃尔玛商品分类 → 预算类别
        """
        budget_to_walmart = cls.get_budget_to_walmart_main()
        return {
            m.lower(): bcat 
            for bcat, mains in budget_to_walmart.items() 
            for m in mains
        }


class SplitRatiosConfig:
    """历史数据转换配置类"""
    
    # ============================================================================
    # 历史数据转换配置（用于将旧格式13类转换为新格式17类）
    # ============================================================================
    # 拆分比例基于 BLS Consumer Expenditure Survey 2022
    # 网址: https://www.bls.gov/cex/
    # ============================================================================
    SPLIT_RATIOS = {
        "low_income": {  # < $35,000/年
            "healthcare_expenditure": {"goods": 0.25, "services": 0.75},
            "transportation_expenditure": {"goods": 0.10, "services": 0.90},
            "education_expenditure": {"goods": 0.15, "services": 0.85}
        },
        "middle_income": {  # $35,000 - $100,000/年
            "healthcare_expenditure": {"goods": 0.18, "services": 0.82},
            "transportation_expenditure": {"goods": 0.08, "services": 0.92},
            "education_expenditure": {"goods": 0.10, "services": 0.90}
        },
        "high_income": {  # > $100,000/年
            "healthcare_expenditure": {"goods": 0.12, "services": 0.88},
            "transportation_expenditure": {"goods": 0.08, "services": 0.92},
            "education_expenditure": {"goods": 0.08, "services": 0.92}
        }
    }
    
    @staticmethod
    def get_income_level(annual_income: float) -> str:
        """
        根据年收入确定收入水平
        
        基于PSID数据集分析结果和美国官方标准：
        - 低收入: < $35,000 (约5.5%家庭，接近联邦贫困线1.5倍)
        - 中等收入: $35,000 - $100,000 (约79.4%家庭，主体人群)
        - 高收入: > $100,000 (约15.1%家庭)
        
        Args:
            annual_income: 年收入金额
            
        Returns:
            str: "low_income", "middle_income", 或 "high_income"
        """
        if annual_income is None or annual_income < 35000:
            return "low_income"
        elif annual_income <= 100000:
            return "middle_income"
        else:
            return "high_income"


class BudgetConfig:
    """
    预算配置统一访问类
    
    提供对所有配置的统一访问接口
    """
    
    # 类别定义
    CATEGORY_KEYS = BudgetCategories.CATEGORY_KEYS
    LEGACY_CATEGORY_KEYS = BudgetCategories.LEGACY_CATEGORY_KEYS
    CATEGORY_NAMES_ZH = BudgetCategories.CATEGORY_NAMES_ZH
    
    # 属性映射
    ATTRIBUTE_TO_CATEGORY_MAPPING = AttributeToCategoryMapping.ATTRIBUTE_TO_CATEGORY_MAPPING
    
    # 商品映射
    BUDGET_TO_WALMART_MAIN = CategoryToWalmartMapping.BUDGET_TO_WALMART_MAIN
    NO_SUBCAT_CATEGORIES = CategoryToWalmartMapping.NO_SUBCAT_CATEGORIES
    
    # 拆分比例
    SPLIT_RATIOS = SplitRatiosConfig.SPLIT_RATIOS
    
    @staticmethod
    def get_budget_to_walmart_main() -> Dict[str, List[str]]:
        """获取预算类别到沃尔玛商品分类的映射（唯一化处理后）"""
        return CategoryToWalmartMapping.get_budget_to_walmart_main()
    
    @staticmethod
    def get_main_to_budget_mapping() -> Dict[str, str]:
        """获取沃尔玛商品分类到预算类别的反向映射"""
        return CategoryToWalmartMapping.get_main_to_budget_mapping()
    
    @staticmethod
    def get_income_level(annual_income: float) -> str:
        """根据年收入确定收入水平"""
        return SplitRatiosConfig.get_income_level(annual_income)

