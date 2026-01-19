/**
 * CopilotKit API Route for LangGraph Integration
 * 
 * 这个文件是前端和 LangGraph Agent 后端的桥梁
 * 负责将前端的请求转发到 LangGraph 服务
 */

import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  langGraphPlatformEndpoint,
} from "@copilotkit/runtime";

/**
 * LangGraph 后端配置
 * 
 * 注意：使用 graphId（不是 assistantId）来指定具体的 graph
 */
const serviceAdapter = langGraphPlatformEndpoint({
  deploymentUrl: process.env.LANGGRAPH_API_URL || "http://127.0.0.1:2024",
  graphId: "economist_agent", // 使用 graphId 指定 graph
});

/**
 * 创建 CopilotRuntime 实例
 * 
 * 当使用 langGraphPlatformEndpoint 时，
 * LangGraph 会处理所有的模型调用和工具执行
 */
const runtime = new CopilotRuntime();

/**
 * POST /api/copilotkit
 * 
 * 接收前端的聊天请求，转发到 LangGraph Agent
 */
export const POST = async (req: Request) => {
  // Debug logging
  console.log("[CopilotKit Route] Incoming request:", req.url);
  console.log("[CopilotKit Route] Method:", req.method);
  
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  try {
    const response = await handleRequest(req);
    console.log("[CopilotKit Route] Response status:", response.status);
    return response;
  } catch (error) {
    console.error("[CopilotKit Route] Error:", error);
    throw error;
  }
};