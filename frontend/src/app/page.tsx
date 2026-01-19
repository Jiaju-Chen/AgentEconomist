"use client";

import { ExperimentDashboard } from "@/components/experiment-dashboard";
import { AgentState, createInitialAgentState } from "@/lib/types";
import {
  useCoAgent,
} from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar, HeaderProps } from "@copilotkit/react-ui";
import {useEffect, useState} from "react";
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
          initial: "Welcome to Agent Economist. My mission is to bridge the gap between your economic intuition and rigorous experimentation.\n\nSimply describe your high-level research idea—whether it's about innovation policy, tax effects, or labor shifts—and I will translate it into an executable simulation. I handle the literature grounding, parameter configuration, and data analysis, so you can focus on the big picture.",
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

  // TODO: 实现从 LangGraph 获取状态的逻辑
  // 目前先显示空状态，等待 Agent 通过对话更新

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen flex justify-center items-center flex-col transition-colors duration-300 overflow-auto"
    >
      <ExperimentDashboard state={state.fs_state} />
    </div>
  );
}
