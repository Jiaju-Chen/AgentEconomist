# AgentEconomist: AI-Powered Economic Research Platform

> 🌐 [中文文档](README_ZH.md) | English

An integrated AI-driven platform for economic policy research through controlled experimentation, combining intelligent agents, multi-agent economic simulation, and academic knowledge retrieval.

## 🎯 Overview

**AgentEconomist** is a comprehensive research platform that bridges economic intuition and rigorous experimentation. It enables researchers to:

- 🤖 **Intelligent Experiment Design**: Automatically design controlled experiments with AI agents
- 📚 **Academic Literature Search**: Retrieve relevant papers using SPECTER2 + Qdrant
- 🏭 **Economic System Simulation**: Run large-scale multi-agent economic simulations
- 📊 **Automated Analysis**: Extract key metrics and generate comparative reports
- 🔬 **Scientific Research**: Support policy impact studies, innovation effects, and more

## ✨ Key Features

### **AgentEconomist Module**
- Modular architecture (prompts, tools, core, state, graph, utils)
- LangGraph-based workflow orchestration
- Next.js frontend with CopilotKit integration
- Real-time experiment monitoring and visualization

### **Economic Simulation**
- Multi-agent system with households, firms, banks, and government
- Dynamic markets: goods, labor, and assets
- Innovation and R&D modeling
- Detailed consumption behavior modeling

### **Knowledge Base**
- 13,000+ indexed academic papers
- SPECTER2 semantic embeddings
- Qdrant vector database
- Advanced semantic search with metadata filtering

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+ (3.11 recommended)
- **Node.js**: 18+ (for frontend)
- **Docker**: For Qdrant vector database
- **Conda**: For environment management (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Jiaju-Chen/AgentEconomist.git
cd AgentEconomist
```

### 2. Environment Setup

#### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n ecosim python=3.11
conda activate ecosim

# Install core dependencies
pip install -r requirements.txt
```

#### Option 2: Using Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
poetry shell
```

### 3. Configure Environment Variables

Create `AgentEconomist/.env` (refer to `AgentEconomist/env.example`):

```bash
# LLM API Configuration (choose one)
OPENAI_API_KEY=your-openai-api-key
# or
DASHSCOPE_API_KEY=your-dashscope-api-key
# or
DEEPSEEK_API_KEY=your-deepseek-api-key

# Optional: Custom endpoint
BASE_URL=https://your-api-endpoint/v1/

# Qdrant Configuration (Knowledge Base)
KB_QDRANT_MODE=remote
KB_QDRANT_HOST=localhost
KB_QDRANT_PORT=6333
```

### 4. Start Qdrant Vector Database

```bash
# Start Qdrant with Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Verify service
curl http://localhost:6333/health
```

### 5. (Optional) Build Knowledge Base Index

If you need literature search functionality:

```bash
cd database
python scripts/build_index.py --incremental
```

### 6. Launch AgentEconomist

#### Backend (LangGraph Server)

```bash
cd AgentEconomist

# Install as editable package
pip install -e .

# Start LangGraph server
langgraph dev
```

The backend will be available at `http://localhost:2024`

#### Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 7. Start Using AgentEconomist

1. Open your browser and navigate to `http://localhost:3000`
2. Describe your research idea in the chat interface
3. The agent will guide you through:
   - Literature review
   - Experiment design
   - Simulation execution
   - Results analysis

## 📁 Project Structure

```
AgentEconomist/
├── AgentEconomist/              # Core Agent Module
│   ├── prompts/                 # System prompts
│   ├── tools/                   # Agent tools
│   │   ├── experiment.py        # Experiment management
│   │   ├── simulation.py        # Simulation execution
│   │   ├── analysis.py          # Data analysis
│   │   ├── knowledge.py         # Literature search
│   │   └── parameter.py         # Parameter management
│   ├── core/                    # Core logic
│   │   ├── manifest.py          # Experiment manifest
│   │   └── yaml_ops.py          # YAML operations
│   ├── state/                   # State management
│   │   ├── types.py             # State type definitions
│   │   └── converter.py         # State converters
│   ├── graph/                   # LangGraph workflow
│   │   ├── agent.py             # Agent graph
│   │   ├── nodes.py             # Graph nodes
│   │   └── state.py             # Graph state
│   ├── utils/                   # Utilities
│   └── config.py                # Configuration
│
├── frontend/                    # Next.js Frontend
│   ├── src/
│   │   ├── app/                 # App routes
│   │   ├── components/          # React components
│   │   └── lib/                 # Utilities
│   └── package.json
│
├── agentsociety_ecosim/         # Economic Simulation Engine
│   ├── agent/                   # Economic agents
│   │   ├── household.py         # Household agents
│   │   ├── firm.py              # Firm agents
│   │   ├── bank.py              # Banking system
│   │   └── government.py        # Government agent
│   ├── center/                  # Economic centers
│   │   ├── ecocenter.py         # Central coordinator
│   │   ├── jobmarket.py         # Labor market
│   │   └── assetmarket.py       # Asset market
│   ├── simulation/              # Simulation engine
│   │   └── simulation.py        # Main simulation loop
│   ├── consumer_modeling/       # Consumer behavior
│   └── mcp_server/              # MCP parameter server
│
├── database/                    # Knowledge Base
│   ├── knowledge_base/          # SPECTER2 + Qdrant
│   │   ├── embeddings.py        # SPECTER2 embeddings
│   │   ├── vector_store.py      # Qdrant integration
│   │   ├── retriever.py         # Semantic search
│   │   └── tool.py              # Agent tool wrapper
│   └── scripts/                 # Indexing scripts
│
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 📖 Usage Guide

### AgentEconomist Workflow

1. **Research Question**: Describe your economic research idea
   - Example: "How does R&D subsidy policy affect innovation and GDP?"

2. **Literature Review**: Agent automatically searches relevant papers
   - Semantic search using SPECTER2 embeddings
   - Results saved to experiment manifest

3. **Experiment Design**: Agent designs controlled experiments
   - Control group (baseline policy)
   - Treatment group (modified policy)
   - YAML configuration files generated

4. **Simulation Execution**: Run multi-agent economic simulation
   - Households make consumption decisions
   - Firms produce goods and innovate
   - Markets clear and adjust prices
   - Government collects taxes and redistributes

5. **Results Analysis**: Extract and compare key metrics
   - Employment rate, GDP, Gini coefficient
   - Innovation events, firm performance
   - Household welfare indicators

6. **Scientific Conclusions**: Draw evidence-based conclusions
   - Statistical significance testing
   - Visualization of trends
   - Actionable recommendations

### Direct Simulation (Without Agent)

```bash
cd agentsociety_ecosim/simulation

# Run with default configuration
python simulation.py

# Run with custom configuration
python simulation.py --config path/to/config.yaml
```

## 🔧 Configuration

### Simulation Parameters

Main configuration file: `default.yaml`

Key parameter categories:

- **Tax Policy**: Income tax rate, VAT rate, corporate tax rate
- **Labor Market**: Dismissal rate, unemployment threshold, dynamic hiring
- **Production**: Labor productivity, labor elasticity, profit conversion
- **Innovation**: Innovation policy, R&D investment ratio
- **System Scale**: Number of households, firms, simulation months

Refer to configuration file comments for detailed parameter descriptions.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API Key | Yes (one of three) |
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API Key | Yes (one of three) |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | Yes (one of three) |
| `BASE_URL` | Custom LLM endpoint | No |
| `KB_QDRANT_MODE` | Qdrant mode (remote/local) | No |
| `KB_QDRANT_HOST` | Qdrant host address | No |
| `KB_QDRANT_PORT` | Qdrant port | No |

## 🛠️ Development

### Running Tests

```bash
# Test simulation
cd agentsociety_ecosim/simulation
python joint_debug_test.py

# Test knowledge base
cd database
python scripts/test_retrieval.py
```

### Code Style

- **Python**: Follow PEP 8 style guide
- **Configuration**: YAML format
- **Documentation**: Markdown format

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Common Issues

### Q: Qdrant connection failed?

**A:** Ensure Qdrant service is running:

```bash
docker ps | grep qdrant
curl http://localhost:6333/health
```

If not running, start Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### Q: How to obtain embedding models?

**A:** Model files are large and need to be downloaded separately:

1. Download `all-MiniLM-L6-v2` from HuggingFace
2. Place in `model/all-MiniLM-L6-v2/` directory
3. Or set `MODEL_PATH` environment variable to model location

### Q: How to configure API keys?

**A:** Create `AgentEconomist/.env` file with:

```bash
OPENAI_API_KEY=your-key-here
BASE_URL=https://your-endpoint/v1/
```

### Q: Where are experiment data saved?

**A:** Experiment configurations and results are saved in `experiment_files/` directory, organized by timestamp and experiment intent.

### Q: Frontend shows "Agent not found" error?

**A:** Ensure:
1. LangGraph server is running on port 2024
2. `AgentEconomist` is installed as a package (`pip install -e .`)
3. `langgraph.json` correctly references the agent graph

## 🏗️ Architecture

### Technology Stack

**Backend:**
- LangGraph 0.6.24 - Agent workflow orchestration
- LangChain - LLM integration
- Python 3.11 - Core language
- Qdrant - Vector database
- SPECTER2 - Academic paper embeddings

**Frontend:**
- Next.js 15 - React framework
- CopilotKit - AI chat interface
- TypeScript - Type safety
- Ant Design - UI components

**Simulation:**
- Ray - Distributed computing
- NumPy/Pandas - Data processing
- Matplotlib - Visualization

### Agent Workflow

```
User Input → System Prompt → LLM → Tool Selection → Tool Execution → State Update → LLM → Response
                ↑                                                            ↓
                └────────────────── Continue or End? ──────────────────────┘
```

Available Tools:
- `get_available_parameters`: List simulation parameters
- `query_knowledge_base`: Search academic literature
- `init_experiment_manifest`: Initialize experiment
- `create_yaml_from_template`: Generate configuration
- `run_simulation`: Execute simulation
- `compare_experiments`: Compare results
- `analyze_experiment_directory`: Analyze outputs

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent workflow framework
- [AgentScope](https://github.com/modelscope/agentscope) - Multi-agent framework
- [SPECTER2](https://github.com/allenai/specter2) - Academic paper embeddings
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ray](https://www.ray.io/) - Distributed computing
- [CopilotKit](https://www.copilotkit.ai/) - AI chat interface

## 📮 Contact

For questions or suggestions, please submit an Issue or Pull Request.

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ for economic research**

**Last Updated**: January 2026
