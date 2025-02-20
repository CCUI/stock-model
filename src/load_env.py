from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment_variables():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found in project root directory")