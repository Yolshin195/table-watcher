import json
import os
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
#  ROIManager
# -----------------------------------------------------------------------
class ROIManager:
    def __init__(self, config_path: str = "settings/table_config.json"):
        self.config_path = Path(config_path)
        self._ensure_dir()

    def _ensure_dir(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def get_roi(self, video_path: str) -> tuple[int, int, int, int]:
        """Возвращает ROI в формате (x, y, w, h)."""
        file_key = Path(video_path).name
        configs = self._load_all_configs()
        
        if file_key in configs:
            logger.info("Загружен сохраненный ROI для: %s", file_key)
            return tuple(configs[file_key])
        
        logger.warning("ROI для '%s' не найден. Требуется ручной выбор.", file_key)
        roi = self._select_interactively(video_path)
        self._save_config(file_key, roi)
        return roi

    def _load_all_configs(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Ошибка чтения конфига ROI: %s", e)
            return {}

    def _save_config(self, file_key: str, roi: tuple):
        configs = self._load_all_configs()
        configs[file_key] = roi
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=4, ensure_ascii=False)

    def _select_interactively(self, video_path: str) -> tuple[int, int, int, int]:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Не удалось открыть видео для выбора ROI: {video_path}")
            
        window_name = f"Select ROI: {Path(video_path).name}"
        # showCrosshair=True помогает точнее целиться
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        # Fix для некоторых ОС, чтобы окно закрылось сразу
        for _ in range(10): cv2.waitKey(1)
        
        x, y, w, h = map(int, roi)
        if w == 0 or h == 0:
            logger.warning("ROI не выбран, используется весь кадр")
            return (0, 0, int(frame.shape[1]), int(frame.shape[0]))
            
        return (x, y, w, h)