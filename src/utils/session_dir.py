"""
utils/session_dir.py

Создаёт уникальную папку для каждого запуска вида:
    outputs/video_name__2024-01-15_14-30-05_a3f2/

Формат имени:
    {video_stem}__{date}_{time}_{uid4}

Пример:
    outputs/
    └── pizzeria_cam1__2024-01-15_14-30-05_a3f2/
        ├── output.mp4
        ├── report.txt
        ├── snapshots/
        └── timeline.png
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
import uuid


def _slugify(name: str) -> str:
    """
    Превращает произвольное имя файла в безопасный slug для папки.
    'My Video (1).mp4' -> 'my_video_1'
    """
    # Убираем расширение если передали имя файла
    stem = Path(name).stem
    # Заменяем все не-буквенно-цифровые символы на подчёркивание
    slug = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE)
    # Убираем ведущие/хвостовые подчёркивания и приводим к нижнему регистру
    return slug.strip("_").lower() or "video"


def get_session_dir(base_output: str, video_path: str = "") -> Path:
    """
    Создаёт и возвращает уникальную папку для текущего запуска.

    Args:
        base_output: корневая папка для всех запусков (например, "outputs")
        video_path:  путь к входному видео — используется для именования папки

    Returns:
        Path к созданной папке запуска, например:
        outputs/pizzeria__2024-01-15_14-30-05_a3f2/
    """
    video_slug = _slugify(video_path) if video_path else "run"
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid        = uuid.uuid4().hex[:6]

    folder_name = f"{video_slug}__{timestamp}_{uid}"
    session_path = Path(base_output) / folder_name
    session_path.mkdir(parents=True, exist_ok=True)

    return session_path