"""
Бизнес-логика мониторинга состояния столика.

Этот модуль не зависит от OpenCV, YOLO или любого другого CV-фреймворка.
На вход принимает только: номер кадра + bool "есть ли человек в зоне".
Вся логика FSM, дебаунс, аналитика — здесь.
"""
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import pandas as pd
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import json

OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

class TableState(Enum):
    """Состояния столика."""
    EMPTY    = auto()   # Стол пустой, никого нет
    OCCUPIED = auto()   # За столом сидит гость
    APPROACH = auto()   # Кто-то подошёл впервые после периода пустоты


@dataclass(frozen=True)
class StateTransition:
    """Факт смены состояния — иммутабельный событийный объект."""
    frame_no:   int
    timestamp:  float          # секунды от начала видео
    prev_state: TableState
    next_state: TableState

    @property
    def event_name(self) -> str:
        return f"{self.prev_state.name} → {self.next_state.name}"


@dataclass
class CleanupRecord:
    """
    Одна запись о цикле «стол освободился → кто-то подошёл».
    Используется для финальной аналитики.
    """
    empty_at_frame: int
    empty_at_sec:   float
    approach_at_frame: Optional[int] = None
    approach_at_sec:   Optional[float] = None

    @property
    def response_time_sec(self) -> Optional[float]:
        """Время реакции (сколько ждали до уборки/нового гостя)."""
        if self.approach_at_sec is None:
            return None
        return self.approach_at_sec - self.empty_at_sec


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class TableMonitor:
    """
    Конечный автомат (FSM) для отслеживания состояния одного столика.

    Принцип работы:
        Вызывай update() на каждом кадре, передавая:
          - frame_no  : порядковый номер кадра
          - fps       : частота кадров видео (для перевода в секунды)
          - occupied  : True если детектор обнаружил человека в зоне ROI

        Класс сам управляет дебаунсом, историей переходов и аналитикой.

    Параметры:
        min_empty_frames  : сколько кадров подряд зона должна быть пустой,
                            чтобы зафиксировать переход в EMPTY.
                            Защищает от мерцания детектора.
        min_occupied_frames: аналогично для перехода в OCCUPIED.
    """

    def __init__(
        self,
        min_empty_frames:    int = 60,   # ~2 сек при 30fps
        min_occupied_frames: int = 30,    # ~1 сек — быстрее реагируем на появление
    ):
        self.min_empty_frames    = min_empty_frames
        self.min_occupied_frames = min_occupied_frames

        # --- Текущее подтверждённое состояние ---
        self._state: TableState = TableState.EMPTY

        # --- Счётчики для дебаунса ---
        self._consecutive_empty:    int = 0
        self._consecutive_occupied: int = 0

        # --- История событий (для DataFrame) ---
        self._transitions: list[StateTransition] = []

        # --- Незакрытые записи об освобождении стола ---
        self._open_cycles: list[CleanupRecord] = []

        # --- Закрытые циклы (есть и empty_at, и approach_at) ---
        self._closed_cycles: list[CleanupRecord] = []

        # --- Полная история
        self.state_history: list[TableState] = []

    # -----------------------------------------------------------------------
    # Публичный API
    # -----------------------------------------------------------------------

    @property
    def state(self) -> TableState:
        """Текущее подтверждённое состояние столика."""
        return self._state

    @property
    def transitions(self) -> list[StateTransition]:
        """Список всех зафиксированных переходов состояний."""
        return list(self._transitions)

    def update(self, frame_no: int, fps: float, occupied: bool) -> Optional[StateTransition]:
        """
        Обработать один кадр.

        Args:
            frame_no : порядковый номер кадра (0-based)
            fps      : частота кадров видео
            occupied : True если в зоне ROI обнаружен человек

        Returns:
            StateTransition если состояние изменилось, иначе None.
        """
        timestamp = frame_no / fps

        if occupied:
            self._consecutive_occupied += 1
            self._consecutive_empty = 0
        else:
            self._consecutive_empty += 1
            self._consecutive_occupied = 0

        transition = self._try_transition(frame_no, timestamp, occupied)

        # Сохраняем текущее состояние в историю после всех расчетов
        self.state_history.append(self._state)

        return transition

    def get_analytics(self) -> dict:
        """
        Итоговая аналитика по всем завершённым циклам.

        Returns:
            Словарь с ключевыми метриками.
        """
        times = [c.response_time_sec for c in self._closed_cycles
                 if c.response_time_sec is not None]

        if not times:
            return {
                "total_cycles":        0,
                "completed_cycles":    0,
                "open_cycles":         len(self._open_cycles),
                "mean_response_sec":   None,
                "median_response_sec": None,
                "min_response_sec":    None,
                "max_response_sec":    None,
            }

        return {
            "total_cycles":        len(self._closed_cycles) + len(self._open_cycles),
            "completed_cycles":    len(self._closed_cycles),
            "open_cycles":         len(self._open_cycles),
            "mean_response_sec":   round(sum(times) / len(times), 2),
            "median_response_sec": round(sorted(times)[len(times) // 2], 2),
            "min_response_sec":    round(min(times), 2),
            "max_response_sec":    round(max(times), 2),
        }

    def get_events_dataframe(self) -> pd.DataFrame:
        """
        Все события в виде Pandas DataFrame.

        Колонки: frame_no, timestamp_sec, prev_state, next_state, event_name
        """
        if not self._transitions:
            return pd.DataFrame(columns=[
                "frame_no", "timestamp_sec", "prev_state", "next_state", "event_name"
            ])

        return pd.DataFrame([
            {
                "frame_no":      t.frame_no,
                "timestamp_sec": round(t.timestamp, 3),
                "prev_state":    t.prev_state.name,
                "next_state":    t.next_state.name,
                "event_name":    t.event_name,
            }
            for t in self._transitions
        ])

    def get_cycles_dataframe(self) -> pd.DataFrame:
        """
        Все циклы «стол освободился → подход» в виде DataFrame.

        Колонки: empty_at_sec, approach_at_sec, response_time_sec, is_completed
        """
        all_cycles = self._closed_cycles + self._open_cycles
        if not all_cycles:
            return pd.DataFrame(columns=[
                "empty_at_frame", "empty_at_sec",
                "approach_at_frame", "approach_at_sec",
                "response_time_sec", "is_completed",
            ])

        return pd.DataFrame([
            {
                "empty_at_frame":    c.empty_at_frame,
                "empty_at_sec":      round(c.empty_at_sec, 3),
                "approach_at_frame": c.approach_at_frame,
                "approach_at_sec":   round(c.approach_at_sec, 3) if c.approach_at_sec else None,
                "response_time_sec": round(c.response_time_sec, 2) if c.response_time_sec else None,
                "is_completed":      c in self._closed_cycles,
            }
            for c in all_cycles
        ])
    
    def get_state_history_dataframe(self, fps: float) -> pd.DataFrame:
        """
        Возвращает полную посекундную историю состояний столика.
        
        Args:
            fps: частота кадров видео для расчета таймстемпов.
        """
        if not self.state_history:
            return pd.DataFrame(columns=["frame_no", "timestamp_sec", "state"])

        return pd.DataFrame([
            {
                "frame_no":      i,
                "timestamp_sec": round(i / fps, 3),
                "state":         s.name
            }
            for i, s in enumerate(self.state_history)
        ])

    # -----------------------------------------------------------------------
    # Внутренняя логика FSM
    # -----------------------------------------------------------------------

    def _try_transition(
        self, frame_no: int, timestamp: float, occupied: bool
    ) -> Optional[StateTransition]:
        """
        Проверить, нужно ли менять состояние.
        Возвращает объект перехода или None.
        """
        new_state = self._resolve_new_state(occupied)

        if new_state is None or new_state == self._state:
            return None

        # Особый случай: EMPTY → OCCUPIED после периода пустоты = APPROACH
        if self._state == TableState.EMPTY and new_state == TableState.OCCUPIED:
            new_state = TableState.APPROACH

        transition = StateTransition(
            frame_no=frame_no,
            timestamp=timestamp,
            prev_state=self._state,
            next_state=new_state,
        )

        self._apply_transition(transition)
        return transition

    def _resolve_new_state(self, occupied: bool) -> Optional[TableState]:
        """
        На основе счётчиков дебаунса определить, хотим ли перейти в новое состояние.
        Возвращает None если ещё не накопилось достаточно кадров.
        """
        if occupied and self._consecutive_occupied >= self.min_occupied_frames:
            return TableState.OCCUPIED
        if not occupied and self._consecutive_empty >= self.min_empty_frames:
            return TableState.EMPTY
        return None

    def _apply_transition(self, t: StateTransition) -> None:
        """Применить переход: обновить состояние и записать аналитику."""
        self._state = t.next_state
        self._transitions.append(t)

        # Стол освободился — открываем новый цикл ожидания
        if t.next_state == TableState.EMPTY:
            self._open_cycles.append(CleanupRecord(
                empty_at_frame=t.frame_no,
                empty_at_sec=t.timestamp,
            ))

        # Кто-то подошёл — закрываем последний открытый цикл
        elif t.next_state == TableState.APPROACH and self._open_cycles:
            cycle = self._open_cycles.pop()
            cycle.approach_at_frame = t.frame_no
            cycle.approach_at_sec   = t.timestamp
            self._closed_cycles.append(cycle)

        # После APPROACH следующий переход — обычный OCCUPIED (не APPROACH снова)
        if t.next_state == TableState.APPROACH:
            self._state = TableState.OCCUPIED


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


# -----------------------------------------------------------------------
#  TableUI
# -----------------------------------------------------------------------
class TableUI:
    """Класс для централизованной отрисовки интерфейса."""
    def __init__(self, polygon: np.ndarray):
        self.polygon = polygon
        # Единая палитра цветов
        self.colors = {
            TableState.EMPTY: (0, 255, 0),      # Зеленый
            TableState.OCCUPIED: (0, 0, 255),   # Красный
            TableState.APPROACH: (0, 255, 255), # Желтый (подход)
        }

    def draw_all(self, frame: np.ndarray, state: TableState, people: list, history: list, total_frames: int):
        """Рисует все слои: ROI, людей и таймлайн."""
        current_color = self.colors.get(state, (255, 255, 255))

        # 1. Рисуем людей
        for person in people:
            color = (0, 0, 255) if person["in_roi"] else (255, 0, 0)
            x1, y1, x2, y2 = person["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, person["point"], 5, color, -1)

        # 2. Рисуем ROI (цветом текущего состояния)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=current_color, thickness=2)
        cv2.putText(frame, f"Status: {state.name}", (self.polygon[0][0], self.polygon[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

        # 3. Рисуем таймлайн поверх видео снизу
        self._draw_timeline(frame, history, total_frames)

    def _draw_timeline(self, frame: np.ndarray, history: list, total_frames: int):
        """Рисует заполняющуюся полоску внизу кадра."""
        h, w = frame.shape[:2]
        bar_h = 15  # Высота полоски в пикселях
        
        if total_frames <= 0: return

        # Масштаб: сколько пикселей занимает один кадр
        step = w / total_frames
        
        # Отрисовка накопленной истории
        for i, s in enumerate(history):
            x_start = int(i * step)
            x_end = int((i + 1) * step)
            color = self.colors.get(s, (100, 100, 100))
            # Рисуем маленький прямоугольник для этого кадра
            cv2.rectangle(frame, (x_start, h - bar_h), (x_end, h), color, -1)


# -----------------------------------------------------------------------
#  VideoProcessor
# -----------------------------------------------------------------------
class VideoProcessor:
    def __init__(
        self, 
        video_path: str, 
        monitor: TableMonitor, 
        polygon: np.ndarray,
        model_variant: str = "yolov8n.pt",
        show_view: bool = True
    ):
        self.video_path = video_path
        self.monitor = monitor
        self.show_view = show_view
        self.polygon = polygon
        
        # Инициализация детектора
        self.model = YOLO(model_variant)
        
        # Видеозахват
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Настройка записи результата (output.mp4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f'{OUTPUT_DIR}/output.mp4', fourcc, self.fps, (self.width, self.height))

        self.ui = TableUI(polygon)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def process(self):
        """Основной цикл обработки."""
        frame_no = 0
        print(f"Обработка началась (Визуализация: {'ВКЛ' if self.show_view else 'ВЫКЛ'})...")

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 1. Анализ
                is_occupied_now, detected_people = self._analyze_frame(frame)

                # 2. Логика
                self.monitor.update(frame_no, self.fps, is_occupied_now)

                # 3. Визуализация и ГЛАВНЫЙ цикл событий UI
                # Мы передаем управление UI-методу, и он говорит нам, пора ли выходить
                should_stop = self._handle_output(frame, self.monitor.state, detected_people)
                
                if should_stop:
                    print("[INFO] Прервано пользователем (нажата 'q').")
                    break

                frame_no += 1
        finally:
            self._cleanup()

    def _analyze_frame(self, frame):
        """Детектирует людей и определяет, кто из них в ROI."""
        results = self.model.predict(frame, classes=[0], verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        
        is_occupied_now = False
        detected_people = []

        for box in boxes:
            # Точка проверки (ноги)
            foot_point = (int((box[0] + box[2]) / 2), int(box[3]))
            
            # Проверяем вхождение в полигон
            in_roi = cv2.pointPolygonTest(self.polygon, foot_point, False) >= 0
            
            if in_roi:
                is_occupied_now = True
            
            # Сохраняем данные для отрисовки
            detected_people.append({
                "box": box.astype(int),
                "point": foot_point,
                "in_roi": in_roi
            })
            
        return is_occupied_now, detected_people

    def _handle_output(self, frame, current_state, detected_people):
        """Вызывает отрисовку всех слоев интерфейса."""
        
        # Используем новый UI менеджер вместо старых cv2.rectangle внутри
        self.ui.draw_all(
            frame, 
            current_state, 
            detected_people, 
            self.monitor.state_history, 
            self.total_frames
        )

        self.out.write(frame)

        if self.show_view:
            cv2.imshow("Monitoring", frame)
            key = cv2.waitKey(1) & 0xFF
            return key == ord('q')
        return False

    def _check_exit_key(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def _cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Обработка завершена. Файл 'output.mp4' сохранен.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Table Cleaning Detection Prototype")
    parser.add_argument("--video", type=str, required=True, help="Путь к видеофайлу")
    parser.add_argument("--headless", action="store_true", help="Запустить без отображения окна")
    args = parser.parse_args()

    # 1. Получаем ROI
    roi_helper = ROIManager()
    table_polygon = roi_helper.get_roi(args.video)

    # 2. Инициализируем бизнес-логику
    # Дебаунс: 30 кадров (~1 сек) для пустоты, 5 кадров для появления
    monitor = TableMonitor(min_empty_frames=30, min_occupied_frames=5)

    # 3. Запускаем процессор
    processor = VideoProcessor(
        video_path=args.video, 
        monitor=monitor,
        polygon=table_polygon,
        show_view=not args.headless
    )
    
    processor.process()

    # 4. Формирование отчета (Требование 2 из ТЗ)
    print("\n" + "="*30)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*30)
    
    analytics = monitor.get_analytics()
    df_events = monitor.get_events_dataframe()
    df_cycles = monitor.get_cycles_dataframe()
    df_state_history = monitor.get_state_history_dataframe()

    print(f"Всего циклов (уход-приход): {analytics['total_cycles']}")
    if analytics['mean_response_sec']:
        print(f"Среднее время реакции: {analytics['mean_response_sec']} сек.")
    else:
        print("Недостаточно данных для расчета среднего времени (никто не подошел к пустому столу).")

    # Сохранение в CSV (Требование ТЗ)
    df_events.to_csv(f"{OUTPUT_DIR}/events_log.csv", index=False)
    df_cycles.to_csv(f"{OUTPUT_DIR}/cleanup_analytics.csv", index=False)
    df_state_history.to_csv(f"{OUTPUT_DIR}/state_history.csv", index=False)
    
    print("\nДетальные логи сохранены в 'events_log.csv' и 'cleanup_analytics.csv'")

if __name__ == "__main__":
    main()