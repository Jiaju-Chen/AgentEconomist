/**
 * CopilotKit API Route for LangGraph Integration
 * 
 * 这个文件是前端和 LangGraph Agent 后端的桥梁
 * 负责将前端的请求转发到 LangGraph 服务
 */

import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphAgent } from "@copilotkit/runtime/langgraph";
import { NextRequest } from "next/server";

// 1. 使用 ExperimentalEmptyAdapter（用于单 agent 场景）
const serviceAdapter = new ExperimentalEmptyAdapter();

// 2. 创建 CopilotRuntime 实例并配置 LangGraph Agent
const runtime = new CopilotRuntime({
  agents: {
    economist_agent: new LangGraphAgent({
      deploymentUrl:
        process.env.LANGGRAPH_API_URL || "http://127.0.0.1:8123",
      graphId: "economist_agent",
      langsmithApiKey: process.env.LANGSMITH_API_KEY || "",
    }),
  },
});

// 3. 构建 Next.js API 路由处理 CopilotKit 请求
export const POST = async (req: NextRequest) => {
  console.log("[CopilotKit Route] Incoming request");
  
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};