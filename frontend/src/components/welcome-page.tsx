"use client";

import { Card, Typography } from "antd";
import { ExperimentOutlined, RocketOutlined, BarChartOutlined } from "@ant-design/icons";

const { Title, Paragraph } = Typography;

export function WelcomePage() {
  return (
    <div 
      style={{ 
        width: "80%", 
        maxWidth: "800px",
        padding: "40px 20px"
      }}
    >
      <Card
        style={{
          borderRadius: "12px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)"
        }}
      >
        {/* 标题 */}
        <div style={{ textAlign: "center", marginBottom: "32px" }}>
          <Title level={1} style={{ margin: 0, fontSize: "48px", fontWeight: "bold" }}>
            AgentEconomist
          </Title>
          <Paragraph style={{ fontSize: "18px", color: "#666", marginTop: "12px" }}>
            From Economic Intuition to Rigorous Simulation
          </Paragraph>
        </div>

        {/* 介绍文本 */}
        <div style={{ marginBottom: "32px" }}>
          <Paragraph style={{ fontSize: "16px", lineHeight: "1.8", textAlign: "justify" }}>
            AgentEconomist bridges the gap between your economic intuition and rigorous experimentation.
            Simply describe your high-level research idea—whether it&apos;s about innovation policy, tax effects, UBI study
            or labor market shifts—and the system will translate it into an executable agent-based simulation.
          </Paragraph>
          <Paragraph style={{ fontSize: "16px", lineHeight: "1.8", textAlign: "justify" }}>
            The agent handles literature grounding, parameter configuration, experimental simulation, data analysis and report generation, 
            allowing you to focus on the big picture while ensuring methodological rigor.
          </Paragraph>
        </div>

        {/* 功能特性 */}
        <div style={{ 
          display: "grid", 
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
          gap: "24px",
          marginTop: "32px"
        }}>
          <FeatureCard 
            icon={<ExperimentOutlined style={{ fontSize: "32px", color: "#1890ff" }} />}
            title="Research Design"
            description="Automatically design experiments based on literature and available parameters"
          />
          <FeatureCard 
            icon={<RocketOutlined style={{ fontSize: "32px", color: "#52c41a" }} />}
            title="Simulation Execution"
            description="Run agent-based economic simulations with configurable parameters"
          />
          <FeatureCard 
            icon={<BarChartOutlined style={{ fontSize: "32px", color: "#faad14" }} />}
            title="Result Analysis"
            description="Compare treatment effects, visualize economic outcomes and generate comprehensive reports"
          />
        </div>

        {/* 开始提示 */}
        <div style={{ 
          marginTop: "40px", 
          padding: "20px", 
          backgroundColor: "#f0f5ff", 
          borderRadius: "8px",
          textAlign: "center"
        }}>
          <Paragraph style={{ margin: 0, fontSize: "16px", color: "#1890ff" }}>
            <strong>Get Started:</strong> Describe your research question in the chat →
          </Paragraph>
        </div>
      </Card>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { 
  icon: React.ReactNode; 
  title: string; 
  description: string;
}) {
  return (
    <div style={{ textAlign: "center", padding: "16px" }}>
      <div style={{ marginBottom: "12px" }}>
        {icon}
      </div>
      <Title level={5} style={{ margin: "8px 0" }}>
        {title}
      </Title>
      <Paragraph style={{ margin: 0, fontSize: "14px", color: "#666" }}>
        {description}
      </Paragraph>
    </div>
  );
}
