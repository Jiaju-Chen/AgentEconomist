/**
 * Experiment Dashboard Component
 * 
 * 显示经济实验的状态、配置、结果图表等
 */

"use client";

import { FSState } from "@/lib/types";
import { Card, Tabs, Tag, Image, Typography, Empty } from "antd";
import { ExperimentOutlined, FileTextOutlined, PictureOutlined, BookOutlined } from "@ant-design/icons";

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

interface ExperimentDashboardProps {
  state: FSState;
}

export function ExperimentDashboard({ state }: ExperimentDashboardProps) {
  // 如果没有实验数据，显示欢迎界面
  if (!state || !state.experiment_id || !state.name) {
    return (
      <Card className="w-full max-w-4xl m-4">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <div>
              <Title level={4}>欢迎使用 Agent Economist</Title>
              <Paragraph>
                请在右侧对话框中描述您的研究想法，我将帮助您：
              </Paragraph>
              <ul className="text-left max-w-md mx-auto">
                <li>📚 检索相关文献支持</li>
                <li>⚙️ 配置仿真参数</li>
                <li>🚀 运行经济实验</li>
                <li>📊 分析实验结果</li>
              </ul>
            </div>
          }
        />
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-6xl m-4">
      {/* 实验基本信息 */}
      <div className="mb-4">
        <Title level={3}>
          <ExperimentOutlined /> {state.name}
        </Title>
        <Paragraph>{state.description}</Paragraph>
        <div className="flex gap-2 mb-2">
          <Tag color="blue">{state.status}</Tag>
          <Tag>{state.created_date}</Tag>
          {state.tags?.map(tag => <Tag key={tag}>{tag}</Tag>)}
        </div>
      </div>

      {/* 标签页 */}
      <Tabs defaultActiveKey="overview">
        {/* 概览 */}
        <TabPane tab="概览" key="overview">
          <div className="space-y-4">
            <div>
              <Text strong>研究问题：</Text>
              <Paragraph>{state.research_question || "未设置"}</Paragraph>
            </div>
            <div>
              <Text strong>假设：</Text>
              <Paragraph>{state.hypothesis || "未设置"}</Paragraph>
            </div>
            <div>
              <Text strong>预期结果：</Text>
              <Paragraph>{state.expected_outcome || "未设置"}</Paragraph>
            </div>
            {state.duration_seconds && (
              <div>
                <Text strong>运行时长：</Text>
                <Text> {Math.round(state.duration_seconds)}秒</Text>
              </div>
            )}
          </div>
        </TabPane>

        {/* 配置 */}
        <TabPane tab={<span><FileTextOutlined /> 配置 ({state.configurations?.length || 0})</span>} key="config">
          <div className="space-y-2">
            {state.configurations && state.configurations.length > 0 ? (
              state.configurations.map((config, idx) => (
                <Card key={idx} size="small">
                  <a href={config.url} target="_blank" rel="noopener noreferrer">
                    📄 {config.filename}
                  </a>
                </Card>
              ))
            ) : (
              <Empty description="暂无配置文件" />
            )}
          </div>
        </TabPane>

        {/* 结果图表 */}
        <TabPane tab={<span><PictureOutlined /> 图表 ({state.images?.length || 0})</span>} key="images">
          <div className="grid grid-cols-2 gap-4">
            {state.images && state.images.length > 0 ? (
              state.images.map((img, idx) => (
                <Card key={idx} size="small" title={img.name}>
                  <Image
                    src={img.url}
                    alt={img.name}
                    fallback="/placeholder-chart.png"
                    preview={{
                      mask: "点击查看大图",
                    }}
                  />
                </Card>
              ))
            ) : (
              <Empty description="暂无实验图表" className="col-span-2" />
            )}
          </div>
        </TabPane>

        {/* 文献支持 */}
        <TabPane tab={<span><BookOutlined /> 文献 ({state.knowledge_base?.length || 0})</span>} key="literature">
          <div className="space-y-2">
            {state.knowledge_base && state.knowledge_base.length > 0 ? (
              state.knowledge_base.map((item, idx) => (
                <Card key={idx} size="small">
                  <div>
                    <Text strong>{item.title}</Text>
                    <br />
                    <Text type="secondary">{item.source}</Text>
                    {item.url && (
                      <>
                        <br />
                        <a href={item.url} target="_blank" rel="noopener noreferrer">
                          🔗 查看原文
                        </a>
                      </>
                    )}
                  </div>
                </Card>
              ))
            ) : (
              <Empty description="暂无文献" />
            )}
          </div>
        </TabPane>
      </Tabs>
    </Card>
  );
}
