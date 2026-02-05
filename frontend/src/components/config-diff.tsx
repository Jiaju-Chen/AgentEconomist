"use client";

import { parseDiff, Diff, Hunk } from "react-diff-view";
import "react-diff-view/style/index.css";
import { Card, Select, Typography, Space, Button } from "antd";
import { useState, useMemo, useEffect, memo } from "react";
import { createPatch } from "diff";

const { Title, Text } = Typography;

interface ConfigDiffProps {
  configurations: {
    filename: string;
    url: string;
  }[];
}

// 使用 diff 库生成 git diff 格式的字符串
function generateDiff(
  oldContent: string, 
  newContent: string, 
  oldFilename: string, 
  newFilename: string,
  expanded: boolean = false
): string {
  // 使用 diff 库的 createPatch 方法生成 unified diff 格式
  // 根据 expanded 参数决定是否显示完整内容
  // context: Infinity 显示所有内容，默认值（通常是3）只显示差异部分
  const patch = createPatch(
    oldFilename,
    oldContent,
    newContent,
    oldFilename, // oldHeader
    newFilename, // newHeader
    { context: expanded ? Infinity : undefined } // options: 展开时显示所有上下文
  );
  
  // createPatch 生成的格式已经是 unified diff，但我们需要确保格式正确
  // 如果 patch 为空（没有差异），我们需要返回一个基本的 diff 头部
  if (!patch || patch.trim() === '') {
    return `diff --git a/${oldFilename} b/${newFilename}\nindex 0000000..1111111 100644\n--- a/${oldFilename}\n+++ b/${newFilename}\n`;
  }
  
  // createPatch 生成的格式已经是标准的 unified diff，可以直接使用
  // 但为了确保兼容性，我们可以添加 git diff 头部（如果还没有的话）
  if (!patch.startsWith('diff --git')) {
    // 如果没有 git diff 头部，添加它
    return `diff --git a/${oldFilename} b/${newFilename}\nindex 0000000..1111111 100644\n${patch}`;
  }
  
  return patch;
}

// 深度比较 configurations 数组的值
function areConfigurationsEqual(
  prevProps: Readonly<ConfigDiffProps>,
  nextProps: Readonly<ConfigDiffProps>
): boolean {
  const prev = prevProps.configurations;
  const next = nextProps.configurations;
  
  if (prev.length !== next.length) {
    return false;
  }
  
  for (let i = 0; i < prev.length; i++) {
    if (prev[i].filename !== next[i].filename || prev[i].url !== next[i].url) {
      return false;
    }
  }
  
  return true;
}

const ConfigDiffComponent = ({ configurations }: ConfigDiffProps) => {
  // 只在有2个或更多配置时显示
  if (configurations.length < 2) {
    return null;
  }

  const [leftIndex, setLeftIndex] = useState<number>(0);
  const [rightIndex, setRightIndex] = useState<number>(1);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [configContents, setConfigContents] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<boolean>(false);

  // 加载配置文件内容
  useEffect(() => {
    const loadConfigs = async () => {
      setLoading(true);
      const contents: Record<string, string> = {};
      
      for (const config of configurations) {
        try {
          // Use API route to fetch file content
          // Add timestamp to bypass browser cache and always get the latest version
          const apiUrl = `/api/files?path=${encodeURIComponent(config.url)}&t=${Date.now()}`;
          const response = await fetch(apiUrl);
          if (response.ok) {
            contents[config.url] = await response.text();
          } else {
            contents[config.url] = `Error loading file: ${response.statusText}`;
          }
        } catch (error) {
          contents[config.url] = `Error loading file: ${error instanceof Error ? error.message : 'Unknown error'}`;
        }
      }
      
      setConfigContents(contents);
      setLoading(false);
    };

    loadConfigs();
  }, [configurations]);

  const diffText = useMemo(() => {
    if (leftIndex >= configurations.length || rightIndex >= configurations.length) return null;
    if (leftIndex === rightIndex) return null;

    const leftConfig = configurations[leftIndex];
    const rightConfig = configurations[rightIndex];

    const leftContent = configContents[leftConfig.url] || "";
    const rightContent = configContents[rightConfig.url] || "";

    if (!leftContent || !rightContent) return null;

    return generateDiff(
      leftContent,
      rightContent,
      leftConfig.filename,
      rightConfig.filename,
      expanded
    );
  }, [configurations, leftIndex, rightIndex, expanded, configContents]);

  const files = useMemo(() => {
    if (!diffText) return [];
    try {
      // parseDiff 的第二个参数可以设置 context 选项
      // 根据 expanded 状态决定是否显示完整内容
      // 注意：类型定义可能不完整，但运行时支持此选项
      return parseDiff(diffText, { context: expanded ? Infinity : undefined } as any);
    } catch (error) {
      console.error("Failed to parse diff:", error);
      return [];
    }
  }, [diffText, expanded]);

  return (
    <div>
      {/* 左侧和右侧下拉框 */}
      <div style={{ display: "flex", gap: 16, marginBottom: 16, alignItems: "center" }}>
        <div style={{ flex: 1 }}>
          <Text strong style={{ marginRight: 8, display: "block", marginBottom: 4 }}>
            Left:
          </Text>
          <Select
            value={leftIndex}
            onChange={setLeftIndex}
            style={{ width: "100%" }}
            options={configurations.map((config, index) => ({
              label: config.filename,
              value: index,
            }))}
          />
        </div>
        <div style={{ flex: 1 }}>
          <Text strong style={{ marginRight: 8, display: "block", marginBottom: 4 }}>
            Right:
          </Text>
          <Select
            value={rightIndex}
            onChange={setRightIndex}
            style={{ width: "100%" }}
            options={configurations.map((config, index) => ({
              label: config.filename,
              value: index,
            }))}
          />
        </div>
      </div>

      {/* Diff 显示区域 */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <Text type="secondary">Loading configurations...</Text>
        </div>
      ) : leftIndex === rightIndex ? (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <Text type="secondary">Please select different configurations to compare</Text>
        </div>
      ) : files.length > 0 ? (
        <>
          <div 
            style={{ 
              border: "1px solid #d9d9d9", 
              borderRadius: "4px", 
              overflow: "hidden",
              maxHeight: expanded ? "600px" : "none",
              overflowY: expanded ? "auto" : "visible",
              overflowX: "auto"
            }}
          >
            {files.map((file, index) => (
              <Diff
                key={`${file.oldRevision}-${file.newRevision}-${index}`}
                viewType="split"
                diffType={file.type}
                hunks={file.hunks}
              >
                {(hunks) =>
                  hunks.map((hunk) => (
                    <Hunk key={hunk.content} hunk={hunk} />
                  ))
                }
              </Diff>
            ))}
          </div>
          {/* Expand/Collapse button */}
          <div style={{ marginTop: 12 }}>
            <Button 
              type="default" 
              onClick={() => setExpanded(!expanded)}
              block
            >
              {expanded ? "Collapse" : "Expand All"}
            </Button>
          </div>
        </>
      ) : (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <Text type="secondary">No differences found</Text>
        </div>
      )}
    </div>
  );
};

// 使用 React.memo 包装组件，只有在 configurations 值变化时才重新渲染
export const ConfigDiff = memo(ConfigDiffComponent, areConfigurationsEqual);

