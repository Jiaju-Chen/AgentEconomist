# Agent实现

在本文件夹中编写各类Agent的实现。

## Household Agent 高级消费决策系统

### `async def consume_advanced(self, product_market: ProductMarket)`

家庭智能体的高级消费决策函数，基于经济学理论和家庭画像进行智能化的消费决策。

#### 功能概述

该函数实现了一个完整的高级消费决策流程，替代了原有的简单LLM消费模式，提供更加科学和个性化的消费行为模拟。

#### 主要特性

1. **预算计算**: 基于经济学公式计算家庭消费预算
   - 使用财富水平、消费倾向和财富指数
   - 公式：`budget = consumption_propensity × (wealth ^ wealth_exponent)`

2. **月度预算分配**: 智能分配预算到不同商品类别
   - 调用`BudgetAllocator.allocate()`进行预算分配
   - 支持12个月的详细预算规划
   - 按类别和子类别进行精细化分配

3. **商品选择**: 基于家庭画像和预算进行商品选择
   - 集成`consumer_decision.py`的商品选择逻辑
   - 使用`build_monthly_shopping_plan()`生成购物计划
   - 支持29,120+商品数据库的智能匹配

4. **商品清单保存**: 自动保存月度购物清单
   - 保存到`consumer_modeling/output/family{household_id}/`目录
   - JSON格式，包含完整的商品信息和预算分配
   - 支持多月份数据累积

#### 执行流程

```
1. 获取当前余额 → 2. 计算消费预算 → 3. 获取/生成月度预算分配
                ↓
8. 降级到简单模式 ← 7. 执行实际购买 ← 6. 保存商品清单 ← 5. 选择商品
    (出错时)                              ↑
                                        4. 获取当前月份预算
```

#### 参数说明

- `product_market: ProductMarket` - 商品市场对象，提供可购买的商品信息

#### 返回值

- 无返回值，但会更新以下状态：
  - `self.purchase_history` - 购买历史记录
  - 家庭余额（通过EconomicCenter更新）
  - 商品市场库存（移除已购买商品）

#### 输出文件

函数会在以下位置生成文件：

1. **月度预算分配文件**:
   - 路径: `consumer_modeling/output/family{household_id}/monthly_subcat_budget_family_{household_id}.json`
   - 内容: 12个月的详细预算分配

2. **月度商品清单文件**:
   - 路径: `consumer_modeling/output/family{household_id}/monthly_shopping_list_family_{household_id}.json`
   - 内容: 每月的具体商品选择和购买计划

#### 依赖模块

- `consumer_modeling.consumer_decision.BudgetAllocator` - 预算分配器
- `consumer_modeling.search_product` - 商品搜索模块
- `consumer_modeling.llm_utils` - LLM工具模块
- `consumer_modeling.family_data` - 家庭数据模块

#### 错误处理

- **模块导入失败**: 自动降级到`consume_simple()`模式
- **预算生成失败**: 记录警告并降级到简单模式
- **商品选择失败**: 记录错误并降级到简单模式
- **购买执行失败**: 记录警告但继续处理其他商品

#### 家庭画像支持

函数支持基于以下家庭特征进行个性化消费：

- 家庭规模 (`family_size`)
- 户主年龄 (`head_age`)
- 户主性别 (`head_gender`)
- 婚姻状况 (`marital_status`)
- 子女数量 (`num_children`)
- 车辆数量 (`num_vehicles`)
- 收入水平 (`income`)
- PSID家庭ID (`psid_family_id`) - 可选

#### 使用示例

```python
# 创建家庭对象（高级消费模式）
household = Household(
    household_id="test_family_001",
    consumption_mode="advanced",
    family_profile={
        "family_size": 4,
        "head_age": 35,
        "income": "middle"
    },
    initial_wealth=15000.0
)

# 执行高级消费
await household.consume_advanced(product_market)
```

#### 性能考虑

- 使用`ThreadPoolExecutor`处理同步/异步兼容性
- 大型商品数据库的内存优化加载
- 智能缓存预算分配结果，避免重复计算

#### 版本兼容性

- 支持降级到`consume_simple()`以保证向后兼容
- 检查高级消费模块可用性，自动适配环境
