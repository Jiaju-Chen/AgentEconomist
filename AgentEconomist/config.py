"""
Configuration management for Agent Economist.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration singleton"""
    
    # Project paths (use relative path calculation)
    _AGENT_DIR = Path(__file__).parent
    PROJECT_ROOT = _AGENT_DIR.parent
    
    SIMULATION_BINARY = PROJECT_ROOT / os.getenv("SIMULATION_BINARY", "agentsociety_ecosim/center/EcoCenter")
    DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "agentsociety_ecosim" / os.getenv("DEFAULT_CONFIG_TEMPLATE", "default.yaml")
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    BASE_URL = os.getenv("ECONOMIST_BASE_URL", "http://35.220.164.252:3888/v1/")
    
    # Model Configuration
    LLM_MODEL = os.getenv("ECONOMIST_LLM_MODEL", "gpt-5")
    LLM_TEMPERATURE = float(os.getenv("ECONOMIST_LLM_TEMPERATURE", "0.1"))
    
    # LangGraph Configuration
    LANGGRAPH_PORT = int(os.getenv("LANGGRAPH_PORT", "8123"))
    LANGGRAPH_HOST = os.getenv("LANGGRAPH_HOST", "0.0.0.0")
    
    # Thread Configuration
    THREAD_ID = os.getenv("ECONOMIST_THREAD_ID", "economist-chat-1")
    
    # Tracing (optional)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in {"true", "1", "yes"}
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agent-economist")
    
    @classmethod
    def get_project_root(cls) -> Path:
        """Get project root directory"""
        return cls.PROJECT_ROOT
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "Missing OPENAI_API_KEY (or DASHSCOPE_API_KEY) environment variable. "
                "Please copy env.example to .env and set your API key."
            )
        
        if not cls.DEFAULT_CONFIG_TEMPLATE.exists():
            raise FileNotFoundError(
                f"Default config template not found: {cls.DEFAULT_CONFIG_TEMPLATE}"
            )
    
    @classmethod
    def get_summary(cls) -> str:
        """Get configuration summary for debugging"""
        return f"""
Agent Economist Configuration:
==============================
Project Root: {cls.PROJECT_ROOT}
Simulation Binary: {cls.SIMULATION_BINARY}
Default Config: {cls.DEFAULT_CONFIG_TEMPLATE}

LLM Model: {cls.LLM_MODEL}
Base URL: {cls.BASE_URL}
Temperature: {cls.LLM_TEMPERATURE}

LangGraph Server: {cls.LANGGRAPH_HOST}:{cls.LANGGRAPH_PORT}
Thread ID: {cls.THREAD_ID}

Tracing Enabled: {cls.LANGCHAIN_TRACING_V2}
"""


# Validate configuration on import
try:
    Config.validate()
    print("✅ Configuration validated successfully")
except Exception as e:
    print(f"⚠️  Configuration validation failed: {e}")
    print("   Please check your .env file and paths.")
