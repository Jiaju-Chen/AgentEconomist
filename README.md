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
