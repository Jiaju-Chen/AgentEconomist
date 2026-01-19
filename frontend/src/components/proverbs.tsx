"use client";

import { AgentState } from "@/lib/types";
import { Card, Tabs, Tag, Typography, Divider, Modal } from "antd";
import { useState, useEffect, useRef } from "react";
import { ConfigDiff } from "./config-diff";

const { Title, Paragraph, Text, Link } = Typography;

export interface ProverbsCardProps {
  state: AgentState;
  setState: (state: AgentState) => void;
}

export function ProverbsCard({ state, setState }: ProverbsCardProps) {
  console.log("rerendering");
  const [activeTab, setActiveTab] = useState<string>("ideas");
  const [enlargedImage, setEnlargedImage] = useState<{ name: string; url: string } | null>(null);
  const prevContentStateRef = useRef<{
    hasIdeas: boolean;
    hasConfig: boolean;
    hasResults: boolean;
  } | null>(null);

  const hasFsState = state.fs_state != null;
  
  // 检查各个标签是否有内容
  const hasIdeas = hasFsState && (
    (state.fs_state.research_question && state.fs_state.research_question.trim() !== "") ||
    (state.fs_state.hypothesis && state.fs_state.hypothesis.trim() !== "") ||
    (state.fs_state.expected_outcome && state.fs_state.expected_outcome.trim() !== "") ||
    (state.fs_state.knowledge_base && state.fs_state.knowledge_base.length > 0)
  );
  
  const hasConfig = hasFsState && 
    state.fs_state.configurations && 
    state.fs_state.configurations.length >= 2;
  
  const hasResults = hasFsState && 
    state.fs_state.images && 
    state.fs_state.images.length > 0;

  // 监听内容变化，自动切换标签
  useEffect(() => {
    if (!prevContentStateRef.current) {
      // 初始化：记录当前状态
      prevContentStateRef.current = {
        hasIdeas,
        hasConfig,
        hasResults,
      };
      return;
    }

    const prev = prevContentStateRef.current;
    
    // 检查哪个标签从无内容变为有内容
    if (!prev.hasIdeas && hasIdeas) {
      setActiveTab("ideas");
    } else if (!prev.hasConfig && hasConfig) {
      setActiveTab("config");
    } else if (!prev.hasResults && hasResults) {
      setActiveTab("results");
    }
    
    // 更新之前的状态
    prevContentStateRef.current = {
      hasIdeas,
      hasConfig,
      hasResults,
    };
  }, [hasIdeas, hasConfig, hasResults]);
  const showRunningTool = state.running_tool_name && state.running_tool_name.trim() !== "";
  const titleText = hasFsState && state.fs_state.name && state.fs_state.name.trim() !== ""
    ? state.fs_state.name
    : "Agent Ecoscientist";
  const hasDescription = hasFsState && state.fs_state.description && state.fs_state.description.trim() !== "";
  const hasCreatedDate = hasFsState && state.fs_state.created_date && state.fs_state.created_date.trim() !== "";

  const cardStyle = hasFsState
    ? {
        width: "80%",
        minHeight: "80%",
        maxHeight: "95vh",
        display: "flex",
        flexDirection: "column" as const,
      }
    : {
        maxHeight: "95vh",
      };

  return (
    <Card
      style={cardStyle}
      styles={{ 
        body: { 
          padding: "24px",
          overflow: "auto",
          flex: 1,
          display: "flex",
          flexDirection: "column",
        },
        header: { padding: "24px" }
      }}
      title={
        <div>
          <Title level={2} style={{ margin: 0, textAlign: "center" }}>
            {titleText}
          </Title>
          {hasCreatedDate && (
            <div style={{ marginTop: 8, textAlign: "center" }}>
              <Paragraph style={{ margin: 0, fontSize: "12px", color: "#999" }}>
                {state.fs_state.created_date}
              </Paragraph>
            </div>
          )}
          {hasDescription && (
            <div style={{ marginTop: 8, textAlign: "left" }}>
              <Paragraph style={{ margin: 0, fontSize: "12px", color: "#666" }}>
                {state.fs_state.description}
              </Paragraph>
            </div>
          )}
        </div>
      }
    >
      {!hasFsState ? (
        <div style={{ textAlign: "center", padding: "40px 20px" }}>
          <Paragraph style={{ fontSize: "16px", color: "#666" }}>
            Welcome to Agent Economist. Please describe the economics research question you would like to investigate.
          </Paragraph>
        </div>
      ) : (
        <>
          {/* 状态区 */}
          <div style={{ marginBottom: 16, padding: "12px", backgroundColor: "#f5f5f5", borderRadius: "4px" }}>
            {state.fs_state.status && (
              <div style={{ marginBottom: showRunningTool ? 8 : 0 }}>
                <Text strong style={{ marginRight: 8 }}>Status: </Text>
                <Tag
                  color={
                    state.fs_state.status === "pending"
                      ? "default"
                      : state.fs_state.status === "planning"
                      ? "cyan"
                      : state.fs_state.status === "running"
                      ? "processing"
                      : state.fs_state.status === "completed"
                      ? "success"
                      : state.fs_state.status === "failed"
                      ? "error"
                      : state.fs_state.status === "analysis_pending"
                      ? "warning"
                      : "default"
                  }
                >
                  {state.fs_state.status}
                </Tag>
              </div>
            )}
            {showRunningTool && (
              <div>
                <Text strong style={{ marginRight: 8 }}>Running Tool: </Text>
                <Tag color="processing">{state.running_tool_name}</Tag>
              </div>
            )}
          </div>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            style={{ 
              display: "flex",
              flexDirection: "column",
              flex: 1,
              overflowY: "auto",
              overflowX: "hidden",
            }}
            items={[
            {
              key: "ideas",
              label: "Experimental Ideas",
              children: (
                <div style={{ padding: "20px 0" }}>
                  {/* 待验证假设部分 */}
                  {hasFsState && (
                    <>
                      {(state.fs_state.research_question || state.fs_state.hypothesis || state.fs_state.expected_outcome) && (
                        <div style={{ marginBottom: 24 }}>
                          <Title level={4}>Hypotheses to Verify</Title>
                          {state.fs_state.research_question && state.fs_state.research_question.trim() !== "" && (
                            <div style={{ marginBottom: 16 }}>
                              <Text strong>Research Question: </Text>
                              <Paragraph style={{ margin: "8px 0 0 0" }}>
                                {state.fs_state.research_question}
                              </Paragraph>
                            </div>
                          )}
                          {state.fs_state.hypothesis && state.fs_state.hypothesis.trim() !== "" && (
                            <div style={{ marginBottom: 16 }}>
                              <Text strong>Hypothesis: </Text>
                              <Paragraph style={{ margin: "8px 0 0 0" }}>
                                {state.fs_state.hypothesis}
                              </Paragraph>
                            </div>
                          )}
                          {state.fs_state.expected_outcome && state.fs_state.expected_outcome.trim() !== "" && (
                            <div style={{ marginBottom: 16 }}>
                              <Text strong>Expected Outcome: </Text>
                              <Paragraph style={{ margin: "8px 0 0 0" }}>
                                {state.fs_state.expected_outcome}
                              </Paragraph>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Knowledge Base 部分 */}
                      {state.fs_state.knowledge_base && state.fs_state.knowledge_base.length > 0 && (
                        <>
                          {((state.fs_state.research_question || state.fs_state.hypothesis || state.fs_state.expected_outcome)) && (
                            <Divider />
                          )}
                          <Title level={4}>Literature</Title>
                          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 16 }}>
                            {state.fs_state.knowledge_base.map((item, index) => (
                              <Card key={index} size="small" style={{ width: "100%" }}>
                                <div style={{ marginBottom: 12 }}>
                                  {item.url ? (
                                    <Link href={item.url} target="_blank" rel="noopener noreferrer">
                                      <Title level={5} style={{ margin: 0 }}>
                                        {item.title}
                                      </Title>
                                    </Link>
                                  ) : (
                                    <Title level={5} style={{ margin: 0 }}>
                                      {item.title}
                                    </Title>
                                  )}
                                </div>
                                {item.source && (
                                  <div style={{ marginBottom: 8 }}>
                                    <Text type="secondary" style={{ fontSize: "12px" }}>
                                      {item.source}
                                    </Text>
                                  </div>
                                )}
                                {item.core_finding && (
                                  <div style={{ marginBottom: 8 }}>
                                    <Text strong>Core Finding: </Text>
                                    <Paragraph style={{ margin: "4px 0 0 0" }}>
                                      {item.core_finding}
                                    </Paragraph>
                                  </div>
                                )}
                                {item.takeaway && (
                                  <div>
                                    <Text strong>Takeaway: </Text>
                                    <Paragraph style={{ margin: "4px 0 0 0" }}>
                                      {item.takeaway}
                                    </Paragraph>
                                  </div>
                                )}
                              </Card>
                            ))}
                          </div>
                        </>
                      )}
                    </>
                  )}
                </div>
              ),
            },
            {
              key: "config",
              label: "Experimental Configuration",
              children: (
                <div style={{ padding: "20px 0" }}>
                  {hasFsState && state.fs_state.configurations && state.fs_state.configurations.length >= 2 ? (
                    <ConfigDiff configurations={state.fs_state.configurations} />
                  ) : (
                    <Paragraph>
                      {hasFsState && state.fs_state.configurations && state.fs_state.configurations.length === 1
                        ? "At least 2 configuration files are required to display the diff."
                        : "No configuration files available."}
                    </Paragraph>
                  )}
                </div>
              ),
            },
            {
              key: "results",
              label: "Experimental Results",
              children: (
                <div style={{ padding: "20px 0" }}>
                  {hasFsState && state.fs_state.images && state.fs_state.images.length > 0 ? (
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "20px",
                        justifyContent: "flex-start",
                      }}
                    >
                      {state.fs_state.images.map((image, index) => {
                        return (
                          <Card
                            key={index}
                            style={{
                              flex: "1 1 calc(33.333% - 14px)",
                              minWidth: "300px",
                              maxWidth: "calc(33.333% - 14px)",
                            }}
                            title={image.name}
                            hoverable
                          >
                            <div
                              style={{
                                width: "100%",
                                aspectRatio: "4 / 3",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                overflow: "hidden",
                                backgroundColor: "#f5f5f5",
                                borderRadius: "4px",
                                cursor: "pointer",
                              }}
                              onClick={() => setEnlargedImage(image)}
                            >
                              <img
                                src={image.url}
                                alt={image.name}
                                style={{
                                  maxWidth: "100%",
                                  maxHeight: "100%",
                                  objectFit: "contain",
                                }}
                                onError={(e) => {
                                  const target = e.target as HTMLImageElement;
                                  target.style.display = "none";
                                  const parent = target.parentElement;
                                  if (parent) {
                                    const errorText = document.createElement("span");
                                    errorText.textContent = "Failed to load image";
                                    errorText.style.color = "#999";
                                    parent.appendChild(errorText);
                                  }
                                }}
                              />
                            </div>
                          </Card>
                        );
                      })}
                    </div>
                  ) : (
                    <Paragraph>
                      No experimental results available.
                    </Paragraph>
                  )}
                  
                  {/* 图片放大 Modal */}
                  <Modal
                    open={enlargedImage !== null}
                    onCancel={() => setEnlargedImage(null)}
                    footer={null}
                    width="90vw"
                    style={{ top: 20 }}
                    styles={{
                      body: {
                        padding: "20px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        minHeight: "80vh",
                        backgroundColor: "#f5f5f5",
                      },
                    }}
                    title={enlargedImage?.name}
                  >
                    {enlargedImage && (
                      <div
                        style={{
                          width: "100%",
                          height: "80vh",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          overflow: "auto",
                        }}
                      >
                        <img
                          src={enlargedImage.url}
                          alt={enlargedImage.name}
                          style={{
                            maxWidth: "100%",
                            maxHeight: "100%",
                            objectFit: "contain",
                          }}
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = "none";
                            const parent = target.parentElement;
                            if (parent) {
                              const errorText = document.createElement("span");
                              errorText.textContent = "Failed to load image";
                              errorText.style.color = "#999";
                              parent.appendChild(errorText);
                            }
                          }}
                        />
                      </div>
                    )}
                  </Modal>
                </div>
              ),
            },
          ]}
          />
        </>
      )}
    </Card>
  );
}
