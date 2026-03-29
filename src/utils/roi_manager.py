import os
import cv2
import json
import numpy as np

# -----------------------------------------------------------------------
#  ROIManager
# -----------------------------------------------------------------------
class ROIManager:
    """
    Класс для управления зонами интереса (ROI).
    Автоматически создает папки и привязывает ROI к конкретным именам файлов.
    """
    def __init__(self, config_path: str = "settings/table_config.json"):
        self.config_path = config_path
        # Обойти ошибку несуществующей папки:
        self._ensure_dir()

    def _ensure_dir(self):
        """Создает папку для конфига, если её нет."""
        dir_name = os.path.dirname(self.config_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"[INFO] Создана директория: {dir_name}")

    def get_roi(self, video_path: str) -> np.ndarray:
        """
        Загружает ROI для конкретного файла. 
        Если файл новый — просит выбрать зону.
        """
        # Используем только имя файла как ключ (чтобы пути /home/user/... не мешали)
        file_key = os.path.basename(video_path)
        
        all_configs = self._load_all_configs()
        
        if file_key in all_configs:
            print(f"[INFO] Найден сохраненный ROI для файла: {file_key}")
            roi = all_configs[file_key]
        else:
            print(f"[WARN] ROI для '{file_key}' не найден. Требуется настройка.")
            roi = self._select_interactively(video_path)
            self._save_config(file_key, roi)

        # Превращаем [x, y, w, h] в полигон
        x, y, w, h = roi
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def _load_all_configs(self) -> dict:
        """Грузит весь JSON файл."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ERROR] Ошибка чтения конфига: {e}")
        return {}

    def _save_config(self, file_key: str, roi: list):
        """Добавляет новый ROI в файл, не стирая старые."""
        configs = self._load_all_configs()
        configs[file_key] = roi
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=4, ensure_ascii=False)
        print(f"[SUCCESS] Настройки для {file_key} сохранены.")

    def _select_interactively(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        window_name = f"Select ROI for: {os.path.basename(video_path)}"
        roi = cv2.selectROI(window_name, frame, fromCenter=False)
        cv2.destroyWindow(window_name)
        for _ in range(10): cv2.waitKey(1)
        
        x, y, w, h = roi
        if w == 0 or h == 0:
            # Fallback: весь кадр, если пользователь нажал Esc
            h_f, w_f = frame.shape[:2]
            return [0, 0, w_f, h_f]
            
        return [int(x), int(y), int(w), int(h)]