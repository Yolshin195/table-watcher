import datetime
from pathlib import Path

def get_session_dir(base_dir: str = "runs") -> Path:
    """Создает уникальную папку для текущего запуска: runs/RUN_20260329_220532"""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = Path(base_dir) / f"run_{now}"
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path