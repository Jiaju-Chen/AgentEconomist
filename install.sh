#!/usr/bin/env bash
# 快速安装脚本 - AgentSociety Economic Simulation
# 使用方法: ./install.sh

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AgentSociety Economic Simulation${NC}"
echo -e "${GREEN}环境安装脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 conda 命令${NC}"
    echo "请先安装 Miniconda 或 Anaconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查 Python 版本
if ! python3 --version &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 python3${NC}"
    exit 1
fi

# 初始化 conda（如果需要）
if ! conda info --envs &> /dev/null; then
    echo -e "${YELLOW}⚠️  初始化 conda...${NC}"
    conda init bash
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 激活 conda base 环境
source "$(conda info --base)/etc/profile.d/conda.sh"

# 检查是否已存在 ecosim 环境
if conda env list | grep -q "^ecosim "; then
    echo -e "${YELLOW}⚠️  Conda 环境 'ecosim' 已存在${NC}"
    read -p "是否删除并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除旧环境...${NC}"
        conda env remove -n ecosim -y
    else
        echo -e "${YELLOW}使用现有环境...${NC}"
        conda activate ecosim
    fi
fi

# 创建 conda 环境（如果不存在）
if ! conda env list | grep -q "^ecosim "; then
    echo -e "${GREEN}📦 创建 Conda 环境 'ecosim' (Python 3.10)...${NC}"
    conda create -n ecosim python=3.10 -y
fi

# 激活环境
echo -e "${GREEN}🔧 激活 Conda 环境...${NC}"
conda activate ecosim

# 验证 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✅ Python 版本: $PYTHON_VERSION${NC}"

# 升级 pip
echo -e "${GREEN}📦 升级 pip...${NC}"
pip install --upgrade pip setuptools wheel

# 安装核心依赖
echo -e "${GREEN}📦 安装核心依赖...${NC}"
pip install agentscope pyyaml ray qdrant-client transformers torch sentence-transformers pandas numpy tqdm

# 安装数据库模块依赖
echo -e "${GREEN}📦 安装数据库模块依赖...${NC}"
cd database
pip install -r requirements.txt
cd ..

# 安装 MCP 服务器依赖
echo -e "${GREEN}📦 安装 MCP 服务器依赖...${NC}"
cd agentsociety_ecosim/mcp_server
pip install -r requirements.txt
cd ../..

# 安装 Streamlit UI 依赖
echo -e "${GREEN}📦 安装 Streamlit UI 依赖...${NC}"
cd economist
pip install -r requirements_ui.txt
cd ..

# 检查 Docker（用于 Qdrant）
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker 已安装${NC}"
    
    # 检查 Qdrant 容器是否运行
    if docker ps | grep -q qdrant; then
        echo -e "${YELLOW}⚠️  Qdrant 容器已在运行${NC}"
    else
        echo -e "${YELLOW}💡 提示: 可以使用以下命令启动 Qdrant:${NC}"
        echo "  docker run -d --name qdrant --network host qdrant/qdrant:latest"
    fi
else
    echo -e "${YELLOW}⚠️  未找到 Docker，Qdrant 将使用本地模式${NC}"
fi

# 创建 .env 文件模板（如果不存在）
if [ ! -f economist/.env ]; then
    echo -e "${GREEN}📝 创建 .env 文件模板...${NC}"
    cat > economist/.env << 'EOF'
# API Keys（至少需要设置一个）
# 选项 1：OpenAI API
OPENAI_API_KEY=your-openai-api-key-here

# 选项 2：阿里云 DashScope API
# DASHSCOPE_API_KEY=your-dashscope-api-key-here

# 选项 3：DeepSeek API（如果使用自定义端点）
# DEEPSEEK_API_KEY=your-deepseek-api-key-here
# BASE_URL=http://your-api-endpoint/v1/

# Qdrant 配置（知识库）
# 如果使用本地 Qdrant（默认）
KB_QDRANT_MODE=local
KB_QDRANT_HOST=localhost
KB_QDRANT_PORT=6333

# 如果使用远程 Qdrant
# KB_QDRANT_MODE=remote
# KB_QDRANT_HOST=your-qdrant-host
# KB_QDRANT_PORT=6333

# 模型路径（可选，默认使用相对路径）
# MODEL_PATH=/root/project/agentsociety-ecosim/model/all-MiniLM-L6-v2
EOF
    echo -e "${YELLOW}⚠️  请编辑 economist/.env 文件，填入你的 API Key${NC}"
else
    echo -e "${GREEN}✅ .env 文件已存在${NC}"
fi

# 验证安装
echo ""
echo -e "${GREEN}🔍 验证安装...${NC}"

# 检查关键包
check_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "installed")
        echo -e "  ${GREEN}✅ $1: $VERSION${NC}"
        return 0
    else
        echo -e "  ${RED}❌ $1: 未安装${NC}"
        return 1
    fi
}

check_package agentscope
check_package qdrant_client
check_package streamlit
check_package transformers
check_package torch

# 设置脚本执行权限
echo ""
echo -e "${GREEN}🔧 设置脚本执行权限...${NC}"
chmod +x economist/run_streamlit.sh
chmod +x economist/run_design_agent.sh
chmod +x economist/run_simulation.sh

# 完成
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ 安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}📝 下一步操作:${NC}"
echo ""
echo "1. 编辑 .env 文件，填入 API Key:"
echo "   nano economist/.env"
echo ""
echo "2. 启动 Qdrant（如果使用 Docker）:"
echo "   docker run -d --name qdrant --network host qdrant/qdrant:latest"
echo ""
echo "3. 构建知识库索引（可选）:"
echo "   cd database"
echo "   python scripts/build_index.py"
echo ""
echo "4. 启动 Streamlit Web 界面:"
echo "   cd economist"
echo "   ./run_streamlit.sh"
echo ""
echo -e "${GREEN}详细文档请参考: INSTALLATION.md${NC}"
echo ""

