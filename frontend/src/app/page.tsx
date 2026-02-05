"use client";

import { ProverbsCard } from "@/components/proverbs";
import { WelcomePage } from "@/components/welcome-page";
import { AgentState, createInitialAgentState } from "@/lib/types";
import { useCoAgent } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar, HeaderProps } from "@copilotkit/react-ui";
import { useState, useEffect } from "react";
import { useChatContext } from "@copilotkit/react-ui";

// 自定义 Header 组件，移除按钮
function CustomHeader({}: HeaderProps) {
  const { labels } = useChatContext();
  
  return (
    <div className="copilotKitHeader">
      <div>{labels.title}</div>
    </div>
  );
}

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  return (
    <main
      style={
        { "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties
      }
    >
      <CopilotSidebar
        defaultOpen={true}
        Header={CustomHeader}
        disableSystemMessage={true}
        clickOutsideToClose={false}
        labels={{
          title: "Popup Assistant",
          initial: "Welcome to AgentEconomist. My mission is to bridge the gap between your economic intuition and rigorous experimentation.\n\nSimply describe your high-level research idea—whether it's about innovation policy, tax effects, or UBI study—and I will translate it into an executable simulation. I handle the literature grounding, parameter configuration, experimental simulation, data analysis and report generation, so you can focus on the big picture.",
        }}
        suggestions={[
          // {
          //   title: "Read Agent State",
          //   message: "What are the proverbs?",
          // },
        ]}
      >
        <YourMainContent themeColor={themeColor} />
      </CopilotSidebar>
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // Initialize with empty state
  const [state, setState] = useState<AgentState>(createInitialAgentState());
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  // 使用 economist_agent 并监听状态变化
  const { state: agentState } = useCoAgent<AgentState>({
    name: "economist_agent",
    initialState: createInitialAgentState(),
  });

  // 当 agent 状态更新时，同步到本地 state
  // 使用 agentState 如果可用，否则使用本地 state
  const currentState = agentState || state;

  // 检测用户是否发送过消息（通过 messages 数组判断）
  useEffect(() => {
    if (currentState.messages && currentState.messages.length > 0) {
      // 检查是否有用户消息（HumanMessage）
      const hasUserMessage = currentState.messages.some((msg: any) => {
        // 检查消息类型，可能是 HumanMessage 或包含 type: 'human'
        return msg._getType?.() === 'human' || msg.type === 'human' || msg.constructor?.name === 'HumanMessage';
      });
      
      if (hasUserMessage && !hasUserInteracted) {
        setHasUserInteracted(true);
      }
    }
  }, [currentState.messages, hasUserInteracted]);

  // 🐛 调试：打印状态信息
  console.log("🔍 Current State:", {
    hasAgentState: !!agentState,
    hasUserInteracted,
    experiment_id: currentState.fs_state?.experiment_id,
    name: currentState.fs_state?.name,
    status: currentState.fs_state?.status,
    manifest_path: currentState.manifest_path,
    running_tool_name: currentState.running_tool_name,
    messages_count: currentState.messages?.length || 0,
    configurations_count: currentState.fs_state?.configurations?.length || 0,
    images_count: currentState.fs_state?.images?.length || 0,
    knowledge_base_count: currentState.fs_state?.knowledge_base?.length || 0,
  });

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen flex justify-center items-center flex-col transition-colors duration-300"
    >
      {!hasUserInteracted ? (
        <WelcomePage />
      ) : (
        <ProverbsCard state={currentState} setState={setState} />
      )}
    </div>
  );
}
