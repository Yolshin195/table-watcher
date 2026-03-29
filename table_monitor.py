"""
Бизнес-логика мониторинга состояния столика.

Этот модуль не зависит от OpenCV, YOLO или любого другого CV-фреймворка.
На вход принимает только: номер кадра + bool "есть ли человек в зоне".
Вся логика FSM, дебаунс, аналитика — здесь.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import pandas as pd


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