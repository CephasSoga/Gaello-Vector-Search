import os
import dotenv

dotenv.load_dotenv(dotenv_path=".env")

def get_env(key: str, default: str = None) -> str:
    return os.getenv(key, default)
