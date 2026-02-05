// State of the agent, make sure this aligns with your agent's state.
// 这个类型必须与后端 AgentEconomist/graph/state.py 中的 AgentState 一致
export type AgentState = {
    messages: any[];                     // LangChain 消息列表
    fs_state: FSAgentState;              // 前端需要的 FSState
    manifest_path: string | null;        // 当前实验的 manifest 路径
    last_tool_output: string | null;     // 最后一次工具调用的输出
    running_tool_name: string | null;    // 当前正在运行的工具名称
  };
  
// State of the FSA (Field Simulation Agent), make sure this aligns with your agent's state.
export type FSAgentState = {
    // ========== 实验基本信息 ==========
    experiment_id: string;              // 实验唯一标识符
    name: string;                       // 实验名称
    description: string;                 // 实验描述
    created_date: string;               // 创建日期 (ISO 8601 格式)
    tags: string[];                      // 标签列表
    
    // ========== 1. 文献支撑 ==========
    // 必须与后端 state/types.py 中的 KnowledgeBaseItem 保持一致
    knowledge_base: {
      title: string;                       // 文献标题
      source: string;                      // 出处（期刊名 + 年份，如 "Nature 2023"）
      url: string | null;                  // 论文链接（PDF/DOI链接）
    }[];
  
    // ========== 实验状态 ==========
    status: "pending" | "planning" | "running" | "completed" | "failed" | "analysis_pending";  // 整体状态
    start_time: string | null;           // 开始时间 (ISO 8601)
    end_time: string | null;             // 结束时间 (ISO 8601)
    duration_seconds: number | null;     // 总耗时（秒）
  
    // ========== 2. 研究想法 ==========
    research_question: string;           // 研究问题
    hypothesis: string;                  // 研究假设
    expected_outcome: string;            // 预期结果
  
    // ========== 4. 实验配置 ==========
  
    configurations: {
      filename: string;  // treatmentA.yaml
      url: string;  // 配置文件URL路径
    }[]
  
    // ========== 实验结果 ==========
    images: {
      name: string;                       // 图片名称/描述
      url: string;                        // 图片URL路径
    }[]
  
    // ========== 5. 文件路径 ==========
    paths: {
      experiment_directory: string;      // 实验目录路径（相对路径）
      manifest_file: string;             // manifest.yaml 路径（相对路径）
      config_files: {
        [groupName: string]: string;     // 组名: 配置文件路径（相对路径）
      };
    };
  };

// 类型别名，保持向后兼容
export type FSState = FSAgentState;

/**
 * 创建初始 Agent 状态
 */
export function createInitialAgentState(): AgentState {
  return {
    messages: [],
    manifest_path: null,
    last_tool_output: null,
    running_tool_name: null,
    fs_state: {
      experiment_id: "",
      name: "",
      description: "",
      created_date: "",
      tags: [],
      knowledge_base: [],
      status: "pending",
      start_time: null,
      end_time: null,
      duration_seconds: null,
      research_question: "",
      hypothesis: "",
      expected_outcome: "",
      configurations: [],
      images: [],
      paths: {
        experiment_directory: "",
        manifest_file: "",
        config_files: {},
      },
    },
  };
}