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
        min_empty_frames:    int = 30,   # ~1 сек при 30fps
        min_occupied_frames: int = 5,    # ~0.17 сек — быстрее реагируем на появление
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
# 
# -----------------------------------------------------------------------

class VideoProcessor:
    def __init__(
        self, 
        video_path: str, 
        monitor: TableMonitor, 
        model_variant: str = "yolov8n.pt",
        show_view: bool = True
    ):
        self.video_path = video_path
        self.monitor = monitor
        self.show_view = show_view
        
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

    def _get_roi(self) -> np.ndarray:
        """Интерактивный выбор зоны столика."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр для выбора ROI")
            
        print("\n[ИНСТРУКЦИЯ] Выделите столик мышкой и нажмите ENTER или SPACE.")
        print("Для отмены нажмите 'c'.")
        
        window_name = "Select Table ROI"
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        
        # --- ИСПРАВЛЕНИЕ ЗАВИСАНИЯ ---
        cv2.destroyWindow(window_name)
        # Нужно "прокрутить" очередь событий несколько раз, чтобы окно закрылось
        for _ in range(10):
            cv2.waitKey(1)
        # -----------------------------
        
        # Сбрасываем видео на начало после выбора
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        x, y, w, h = roi
        if w == 0 or h == 0:
            print("Предупреждение: ROI не выбран, использую всё поле кадра.")
            return np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
            
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def process(self):
        """Основной цикл обработки."""
        polygon = self._get_roi()
        frame_no = 0

        print(f"Обработка началась (Визуализация: {'ВКЛ' if self.show_view else 'ВЫКЛ'})...")

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 1. Детекция людей (class 0 в COCO - это person)
                results = self.model.predict(frame, classes=[0], verbose=False)[0]
                
                # Проверяем, есть ли хотя бы один человек в зоне ROI
                is_occupied_now = False
                boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
                
                for box in boxes:
                    # Центр нижней линии bbox (ноги человека) — самая точная точка для вхождения в зону
                    center_bottom = (int((box[0] + box[2]) / 2), int(box[3]))
                    
                    if cv2.pointPolygonTest(polygon, center_bottom, False) >= 0:
                        is_occupied_now = True
                        break # Нам достаточно одного человека

                # 2. Обновление бизнес-логики (твой TableMonitor)
                self.monitor.update(frame_no, self.fps, is_occupied_now)

                # 3. Визуализация
                current_state = self.monitor.state
                # Цвета: EMPTY - Зеленый, OCCUPIED/APPROACH - Красный
                color = (0, 255, 0) if current_state == TableState.EMPTY else (0, 0, 255)
                
                # Рисуем зону стола
                cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
                
                # Текст состояния
                label = f"Status: {current_state.name}"
                cv2.putText(frame, label, (polygon[0][0], polygon[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Запись в файл
                self.out.write(frame)

                # Отображение на экране
                if self.show_view:
                    cv2.imshow("Monitoring", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_no += 1

        finally:
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

    # 1. Инициализируем бизнес-логику
    # Дебаунс: 30 кадров (~1 сек) для пустоты, 5 кадров для появления
    monitor = TableMonitor(min_empty_frames=30, min_occupied_frames=5)

    # 2. Запускаем процессор
    processor = VideoProcessor(
        video_path=args.video, 
        monitor=monitor, 
        show_view=not args.headless
    )
    
    processor.process()

    # 3. Формирование отчета (Требование 2 из ТЗ)
    print("\n" + "="*30)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*30)
    
    analytics = monitor.get_analytics()
    df_events = monitor.get_events_dataframe()
    df_cycles = monitor.get_cycles_dataframe()

    print(f"Всего циклов (уход-приход): {analytics['total_cycles']}")
    if analytics['mean_response_sec']:
        print(f"Среднее время реакции: {analytics['mean_response_sec']} сек.")
    else:
        print("Недостаточно данных для расчета среднего времени (никто не подошел к пустому столу).")

    # Сохранение в CSV (Требование ТЗ)
    df_events.to_csv(f"{OUTPUT_DIR}/events_log.csv", index=False)
    df_cycles.to_csv(f"{OUTPUT_DIR}/cleanup_analytics.csv", index=False)
    print("\nДетальные логи сохранены в 'events_log.csv' и 'cleanup_analytics.csv'")

if __name__ == "__main__":
    main()