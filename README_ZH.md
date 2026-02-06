# AgentEconomist

AgentEconomist 通过基于主体的仿真实验，连接经济直觉与严谨实验。

## 环境配置

### Python 环境（Conda - 推荐）

```bash
# 创建 conda 环境（需要 Python 3.11+）
conda create -n economist python=3.11
conda activate economist

# 安装 PyTorch (CPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装依赖
pip install -r requirements.txt
```

### Python 环境（venv - 备选）

```bash
# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装 PyTorch (CPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装依赖
pip install -r requirements.txt
```

### Node.js 环境

```bash
cd frontend
npm install
```

## 数据准备

以下大型数据文件未包含在仓库中，需要手动准备：

### 1. 预训练模型

将预训练模型放置在 `model/` 目录下：

```bash
mkdir -p model
# 将你的模型文件复制到 model/ 目录
```

### 2. 学术论文知识库

将论文数据放置在 `database/Crawl_Results/` 目录下。预期目录结构：

```
database/Crawl_Results/
├── Articles/              # 自然科学/综合类期刊（Nature 系列等）
│   └── [期刊名称]/
│       └── [论文ID]/
│           └── article.json
└── Articles-Social/       # 社会科学期刊（AER, AJS 等）
    └── [期刊名称]/
        └── [论文ID]/
            └── article.json
```

然后使用 SPECTER2 嵌入构建向量索引：

```bash
cd database

# 构建索引
python scripts/build_index.py

# 增量索引（跳过已索引的论文）
python scripts/build_index.py --incremental
```

详细说明请参考 [database/README.md](database/README.md)。

### 3. 仿真系统数据

将仿真数据文件放置在以下目录：

```bash
# 主要仿真数据
mkdir -p agentsociety_ecosim/data
# 将你的仿真数据文件复制到 agentsociety_ecosim/data/

# 家庭建模数据
mkdir -p agentsociety_ecosim/consumer_modeling/household_data
# 将你的家庭数据文件复制到 agentsociety_ecosim/consumer_modeling/household_data/
```

## 部署

### 后端

```bash
cd AgentEconomist
bash start_agent.sh
```

后端运行在 `http://127.0.0.1:8123`

### 前端

```bash
cd frontend
npm run dev
```

前端运行在 `http://localhost:3001`
