# 环境配置与 Qdrant 启动指南

## 环境配置

### Conda 环境
```bash
# 激活 ecosim 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosim
```

### 环境变量
确保以下环境变量已设置：
- `MODEL_PATH`: Embedding 模型路径（默认: `all-MiniLM-L6-v2`）
- `PYTHONPATH`: 包含项目根目录

## Qdrant 启动

### 方式一：Docker（推荐）
```bash
# 使用 host 网络模式启动（避免端口映射权限问题）
docker run -d --name qdrant --network host qdrant/qdrant:latest

# 验证服务
curl http://localhost:6333/health
```

### 方式二：后台运行（临时容器）
```bash
docker run --rm --network host qdrant/qdrant:latest &
```

### 停止 Qdrant
```bash
docker stop qdrant
# 或停止所有 Qdrant 容器
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant:latest)
```

## 验证

运行仿真前，确保：
1. ✅ Conda 环境已激活（`ecosim`）
2. ✅ Qdrant 服务运行中（端口 6333）
3. ✅ 环境变量已配置

检查命令：
```bash
# 检查环境
conda info --envs | grep "*"
python --version

# 检查 Qdrant
curl http://localhost:6333/health
```

