"""
Обработчик видеопотока с плагин-архитектурой.

Класс VideoProcessor открывает видео, кадр за кадром прогоняет его через
детектор людей и TableMonitor, затем передаёт результат всем зарегистрированным
плагинам. Если плагинов нет — видео обрабатывается в фоне без отрисовки.

Плагин — любой объект, унаследованный от BasePlugin.
Плагины регистрируются двумя способами:
    1. Списком в конструкторе: VideoProcessor(plugins=[MyPlugin()])
    2. Методом:              processor.add_plugin(MyPlugin())
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from table_monitor import TableMonitor, TableState, StateTransition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Контекст кадра — единственный объект, передаваемый в плагин
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonDetection:
    """Результат детекции одного человека на кадре."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    foot_point: tuple[int, int]      # (x, y) - точка для проверки в ROI
    is_in_roi: bool                  # попал ли в зону
    confidence: float                # уверенность модели

@dataclass
class FrameContext:
    """
    Всё что плагин может знать о текущем кадре.
    Плагин НЕ должен хранить ссылку на frame после возврата из on_frame() —
    массив переиспользуется VideoCapture.
    """
    frame:      np.ndarray          # сырой BGR-кадр (можно рисовать поверх)
    frame_no:   int                 # порядковый номер кадра (0-based)
    fps:        float               # частота кадров источника
    roi:        tuple[int,int,int,int]  # (x, y, w, h) зоны столика
    state:      TableState          # текущее подтверждённое состояние FSM
    transition: Optional[StateTransition]  # не None если состояние только что изменилось
    occupied:   bool                # детектор нашёл человека в ROI в этом кадре
    detected_people: list[PersonDetection] = field(default_factory=list)

    @property
    def timestamp_sec(self) -> float:
        return self.frame_no / self.fps

    @property
    def roi_rect(self) -> tuple[int,int,int,int]:
        """(x1, y1, x2, y2) — удобно для cv2.rectangle."""
        x, y, w, h = self.roi
        return x, y, x + w, y + h


# ---------------------------------------------------------------------------
# Базовый класс плагина
# ---------------------------------------------------------------------------

class BasePlugin(abc.ABC):
    """
    Интерфейс плагина. Наследуй и переопредели нужные методы.

    Жизненный цикл:
        on_start()  — вызывается один раз перед первым кадром
        on_frame()  — вызывается на каждом кадре
        on_finish() — вызывается после последнего кадра (или при ошибке)
    """

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        """Инициализация плагина. Вызывается до начала обработки."""
        pass

    @abc.abstractmethod
    def on_frame(self, ctx: FrameContext) -> None:
        """Обработать один кадр. Можно рисовать на ctx.frame."""

    def on_finish(self, monitor: TableMonitor) -> None:
        """
        Финализация. Вызывается после всех кадров.
        Получает полный TableMonitor с аналитикой.
        """


# ---------------------------------------------------------------------------
# Основной класс процессора
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Открывает видеофайл и обрабатывает его кадр за кадром.

    Детекция людей выполняется через YOLOv8n (Ultralytics).
    Бизнес-логика делегируется в TableMonitor.
    Отрисовка и любой side-effect — в плагины.

    Args:
        video_path:          путь к видеофайлу
        roi:                 (x, y, w, h) зона столика; если None — запросит у пользователя
        monitor:             экземпляр TableMonitor; если None — создаётся с дефолтными параметрами
        plugins:             список плагинов, подключаемых сразу
        confidence_threshold: минимальная уверенность детектора (0.0–1.0)
        output_path:         путь для записи обработанного видео; None — не писать
        model_name:          имя YOLO-модели (yolov8n.pt / yolov8s.pt / …)
    """

    def __init__(
        self,
        video_path:           str,
        roi:                  Optional[tuple[int,int,int,int]] = None,
        monitor:              Optional[TableMonitor] = None,
        plugins:              Optional[list[BasePlugin]] = None,
        confidence_threshold: float = 0.4,
        output_path:          Optional[str] = None,
        model_name:           str = "yolov8n.pt",
        detection_step:       int = 1,
    ):
        self.video_path           = video_path
        self.roi                  = roi
        self.monitor              = monitor or TableMonitor()
        self.confidence_threshold = confidence_threshold
        self.output_path          = output_path
        self.model_name           = model_name
        self.detection_step       = max(1, detection_step)

        self._plugins: list[BasePlugin] = []
        for p in (plugins or []):
            self.add_plugin(p)

        self._model         = None   # загружается лениво при первом run()
        self._last_occupied = False  # кэш последнего результата детекции
        self._last_detections = None # кэш последних определенных людей

    # -----------------------------------------------------------------------
    # Регистрация плагинов
    # -----------------------------------------------------------------------

    def add_plugin(self, plugin: BasePlugin) -> "VideoProcessor":
        """Добавить плагин. Можно вызывать до и после run(). Возвращает self."""
        self._plugins.append(plugin)
        return self

    # -----------------------------------------------------------------------
    # Запуск обработки
    # -----------------------------------------------------------------------

    def run(self) -> TableMonitor:
        """
        Запустить обработку видео.

        Returns:
            TableMonitor с полной историей переходов и аналитикой.
        """
        cap = self._open_capture()
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Определяем ROI если не задан
        if self.roi is None:
            self.roi = self._select_roi(cap)

        # Инициализируем модель
        self._model = self._load_model()

        # Инициализируем запись выходного видео
        writer = self._make_writer(fps, frame_w, frame_h) if self.output_path else None

        # Уведомляем плагины о старте
        for plugin in self._plugins:
            plugin.on_start(total_frames=total, fps=fps, roi=self.roi)

        logger.info("Начало обработки: %s (%d кадров, %.1f fps)", self.video_path, total, fps)

        frame_no = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- Детекция (запускаем YOLO только каждые detection_step кадров) ---
                if frame_no % self.detection_step == 0:
                    self._last_occupied, self._last_detections = self._detect_person_in_roi(frame, self.roi)
                occupied = self._last_occupied
                detections = self._last_detections

                # --- FSM ---
                transition = self.monitor.update(frame_no, fps, occupied)

                # --- Контекст для плагинов ---
                ctx = FrameContext(
                    frame=frame,
                    frame_no=frame_no,
                    fps=fps,
                    roi=self.roi,
                    state=self.monitor.state,
                    transition=transition,
                    occupied=occupied,
                    detected_people=detections,
                )

                # --- Плагины рисуют / логируют ---
                for plugin in self._plugins:
                    plugin.on_frame(ctx)

                # --- Пишем кадр в файл (уже с отрисовкой плагинов) ---
                if writer is not None:
                    writer.write(frame)

                frame_no += 1

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        logger.info("Обработка завершена. Кадров: %d", frame_no)

        # Финализируем плагины
        for plugin in self._plugins:
            plugin.on_finish(self.monitor)

        return self.monitor

    # -----------------------------------------------------------------------
    # Внутренние методы
    # -----------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {self.video_path}")
        return cap

    def _select_roi(self, cap: cv2.VideoCapture) -> tuple[int,int,int,int]:
        """Попросить пользователя выбрать ROI на первом кадре."""
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Не удалось прочитать первый кадр для выбора ROI")

        logger.info("Выберите зону столика мышью и нажмите ENTER или SPACE")
        roi = cv2.selectROI("Выберите столик", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Выберите столик")

        # Перемотать на начало
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        x, y, w, h = roi
        if w == 0 or h == 0:
            raise ValueError("ROI не выбран (нулевой размер)")

        logger.info("ROI выбран: x=%d y=%d w=%d h=%d", x, y, w, h)
        return int(x), int(y), int(w), int(h)

    def _load_model(self):
        """Загрузить YOLO-модель. Импорт отложен — не ломает тесты без ultralytics."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_name)
            logger.info("Загружена модель: %s", self.model_name)
            return model
        except ImportError:
            raise ImportError(
                "Установи ultralytics: pip install ultralytics"
            )

    def _detect_person_in_roi(
        self, frame: np.ndarray, roi: tuple[int, int, int, int]
    ) -> tuple[bool, list[PersonDetection]]:
        rx, ry, rw, rh = roi
        roi_x2, roi_y2 = rx + rw, ry + rh

        results = self._model(frame, verbose=False, classes=[0])
        
        is_occupied_now = False
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Точка проверки (середина нижней границы bbox - "ноги")
            foot_point = ((x1 + x2) // 2, y2)
            
            # Проверка попадания в прямоугольник ROI
            in_roi = (rx <= foot_point[0] <= roi_x2 and ry <= foot_point[1] <= roi_y2)
            
            if in_roi:
                is_occupied_now = True
                
            detections.append(PersonDetection(
                bbox=(x1, y1, x2, y2),
                foot_point=foot_point,
                is_in_roi=in_roi,
                confidence=conf
            ))
            
        return is_occupied_now, detections

    def _make_writer(
        self, fps: float, width: int, height: int
    ) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать VideoWriter: {self.output_path}")
        logger.info("Запись видео: %s", self.output_path)
        return writer