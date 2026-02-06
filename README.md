# AgentEconomist

AgentEconomist bridges economic intuition and rigorous experimentation through agent-based simulation.

## Environment Setup

### Python Environment (Conda - Recommended)

```bash
# Create conda environment (Python 3.11+ required)
conda create -n economist python=3.11
conda activate economist

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

### Python Environment (venv - Alternative)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

### Node.js Environment

```bash
cd frontend
npm install
```

## Data Preparation

The following large data files are not included in the repository. You need to prepare them manually:

### 1. Pre-trained Models

Place your pre-trained models in the `model/` directory:

```bash
mkdir -p model
# Copy your model files to model/
```

### 2. Academic Paper Knowledge Base

Place your paper data in `database/Crawl_Results/` directory. The expected structure:

```
database/Crawl_Results/
├── Articles/              # Natural science/comprehensive journals (Nature series, etc.)
│   └── [journal_name]/
│       └── [paper_id]/
│           └── article.json
└── Articles-Social/       # Social science journals (AER, AJS, etc.)
    └── [journal_name]/
        └── [paper_id]/
            └── article.json
```

Then build the vector index using SPECTER2 embeddings:

```bash
cd database

# Build the index
python scripts/build_index.py

# For incremental indexing (skip already indexed papers)
python scripts/build_index.py --incremental
```

For detailed instructions, see [database/README.md](database/README.md).

### 3. Simulation Data

Place simulation data files in the following directories:

```bash
# Main simulation data
mkdir -p agentsociety_ecosim/data
# Copy your simulation data files to agentsociety_ecosim/data/

# Household modeling data
mkdir -p agentsociety_ecosim/consumer_modeling/household_data
# Copy your household data files to agentsociety_ecosim/consumer_modeling/household_data/
```

## Deployment

### Backend

```bash
cd AgentEconomist
bash start_agent.sh
```

Backend runs on `http://127.0.0.1:8123`

### Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on `http://localhost:3001`
