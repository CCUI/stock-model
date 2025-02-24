from pathlib import Path
from dotenv import load_dotenv
import os

def load_environment_variables():
    """Load environment variables from .env file and validate required variables"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found in project root directory")
    
    # Validate required environment variables
    required_vars = ['NEWS_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}\nPlease ensure these are set in your .env file.")