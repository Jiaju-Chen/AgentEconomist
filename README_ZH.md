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
