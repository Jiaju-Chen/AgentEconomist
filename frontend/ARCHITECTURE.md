# Frontend 代码架构说明

## 📁 目录结构

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router 目录
│   │   ├── api/               # API 路由
│   │   │   └── copilotkit/
│   │   │       └── route.ts  # CopilotKit API 端点（连接后端）
│   │   ├── layout.tsx         # 根布局（包装 CopilotKit Provider）
│   │   ├── page.tsx           # 主页面（聊天界面 + 实验仪表板）
│   │   ├── globals.css        # 全局样式
│   │   └── favicon.ico        # 网站图标
│   │
│   └── components/            # React 组件
│       ├── experiment-dashboard.tsx  # 实验仪表板（显示实验状态、结果）
│       ├── config-diff.tsx           # 配置差异对比组件
│       ├── proverbs.tsx              # 实验信息卡片（研究问题、假设等）
│       ├── moon.tsx                 # 装饰性组件
│       └── weather.tsx               # 装饰性组件
│
├── agent/                      # LangGraph Agent 配置
│   ├── langgraph.json         # LangGraph 配置文件
│   └── ...
│
├── public/                     # 静态资源
├── scripts/                    # 脚本文件
│   ├── setup-agent.sh         # Agent 环境设置脚本
│   └── setup-agent.bat        # Windows 版本
│
├── package.json                # 依赖和脚本配置
├── tsconfig.json               # TypeScript 配置
├── next.config.ts              # Next.js 配置
└── .npmrc                      # npm 镜像配置
```

## 🏗️ 架构层次

### 1. **框架层：Next.js 16 + React 19**

- **Next.js App Router**：使用最新的 App Router 架构
- **TypeScript**：全类型安全
- **Tailwind CSS**：样式框架
- **Ant Design**：UI 组件库

### 2. **AI 集成层：CopilotKit**

```
用户输入
  ↓
CopilotSidebar (UI 组件)
  ↓
CopilotKit Provider (layout.tsx)
  ↓
/api/copilotkit (Next.js API Route)
  ↓
LangGraph Backend (:8123)
```

**关键文件：**
- `src/app/layout.tsx`：包装 `CopilotKit` Provider
- `src/app/api/copilotkit/route.ts`：API 路由，转发请求到后端
- `src/app/page.tsx`：使用 `CopilotSidebar` 组件

### 3. **组件层：React 组件**

#### 核心组件

1. **`CopilotSidebar`** (`page.tsx`)
   - 聊天侧边栏
   - 接收用户输入
   - 显示 Agent 回复

2. **`ExperimentDashboard`** (`components/experiment-dashboard.tsx`)
   - 显示实验状态、配置、结果
   - 标签页：概览、配置、结果、文献
   - 使用 Ant Design 组件

3. **`ConfigDiff`** (`components/config-diff.tsx`)
   - 对比实验配置差异（Control vs Treatment）
   - 使用 `react-diff-view` 显示差异

4. **`ProverbsCard`** (`components/proverbs.tsx`)
   - 显示研究问题、假设、预期结果
   - 显示实验配置和结果图表

## 🔄 数据流

### 用户交互流程

```
1. 用户在 CopilotSidebar 输入问题
   ↓
2. 请求发送到 /api/copilotkit
   ↓
3. route.ts 转发到 LangGraph Backend (http://127.0.0.1:8123)
   ↓
4. LangGraph Agent 处理请求：
   - 调用工具（文献搜索、参数配置、仿真执行等）
   - 更新 Agent 状态
   ↓
5. 响应返回前端
   ↓
6. CopilotKit 更新 UI
   ↓
7. ExperimentDashboard 显示最新状态
```

### 状态管理

**当前实现：**
- 使用 React `useState` 管理本地状态
- Agent 状态通过 CopilotKit 的 `useCoAgent` Hook 获取
- `page.tsx` 中的 `state` 存储 `AgentState`

**注意：** `@/lib/types` 文件缺失，需要创建类型定义文件。

## 📦 依赖说明

### 核心依赖

1. **CopilotKit** (`@copilotkit/*`)
   - `react-core`：核心功能
   - `react-ui`：UI 组件
   - `runtime`：运行时（API 路由使用）

2. **Next.js 16**
   - App Router
   - Server Components / Client Components
   - API Routes

3. **Ant Design 6**
   - UI 组件库
   - 提供 Card、Tabs、Tag 等组件

4. **React 19**
   - 最新版本的 React

### 工具库

- `react-diff-view`：显示代码差异
- `diff`：生成差异对比
- `concurrently`：同时运行多个命令

## 🔧 配置文件

### `next.config.ts`

```typescript
{
  serverExternalPackages: ["@copilotkit/runtime"]
}
```

- 将 `@copilotkit/runtime` 标记为外部包（不打包）

### `tsconfig.json`

- 路径别名：`@/*` → `./src/*`
- 使用 Next.js 插件

### `.npmrc`

```
registry=https://registry.npmmirror.com
```

- 配置淘宝镜像加速下载

### `package.json` Scripts

```json
{
  "dev": "npm run dev:ui",                    // 只启动前端
  "dev:with-agent": "concurrently ...",      // 同时启动前后端
  "dev:ui": "next dev --turbopack -p 3001",  // 前端开发服务器
  "dev:agent": "conda run -n economist ...", // 后端开发服务器
  "build": "next build",                      // 构建生产版本
  "start": "next start"                       // 启动生产服务器
}
```

## 🚀 启动流程

### 开发模式

```bash
# 方式 1：只启动前端
npm run dev
# → 前端运行在 http://localhost:3001
# → 需要单独启动后端

# 方式 2：同时启动前后端
npm run dev:with-agent
# → 前端：http://localhost:3001
# → 后端：http://localhost:8123
```

### 生产模式

```bash
npm run build  # 构建
npm start      # 启动生产服务器
```

## 🔌 后端连接

### API 端点配置

**文件：** `src/app/api/copilotkit/route.ts`

```typescript
const serviceAdapter = langGraphPlatformEndpoint({
  deploymentUrl: process.env.LANGGRAPH_API_URL || "http://127.0.0.1:8123",
  graphId: "economist_agent",
});
```

**环境变量：**
- `LANGGRAPH_API_URL`：后端地址（默认 `http://127.0.0.1:8123`）
- 可通过 `frontend/.env.local` 配置

## 📝 待完善功能

1. **类型定义缺失**
   - `@/lib/types.ts` 文件不存在
   - 需要创建 `AgentState`、`FSState` 等类型定义

2. **状态同步**
   - `page.tsx` 中 TODO：实现从 LangGraph 获取状态的逻辑
   - 当前使用本地状态，需要与后端状态同步

3. **错误处理**
   - API 路由需要更完善的错误处理
   - 前端需要错误边界组件

## 🎨 UI 设计

- **主色调**：`#6366f1` (Indigo)
- **布局**：侧边栏聊天 + 主内容区
- **组件库**：Ant Design 6
- **样式**：Tailwind CSS + CSS Variables

## 📚 相关文档

- [Next.js 文档](https://nextjs.org/docs)
- [CopilotKit 文档](https://docs.copilotkit.ai/)
- [Ant Design 文档](https://ant.design/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
