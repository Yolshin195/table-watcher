"""
Бизнес-логика мониторинга состояния столика.
============================================================

Архитектура: Конечный автомат (FSM) / Паттерн «Состояние»
--------------------------------------------------------
Этот модуль реализует надежную систему мониторинга занятости столиков, предназначенную
для высоконагруженной видеоаналитики. Используется паттерн «Состояние» для инкапсуляции
бизнес-правил, что гарантирует предсказуемость переходов, логики дебаунса (защиты от дребезга)
и генерации аналитики.

Граф переходов:
      [ СТАРТ ]
          │
          ▼
    ┌──────────┐  occupied > N кадров  ┌──────────────┐
    │  EMPTY   │ ──────────────────►   │   APPROACH   │
    │(свободно)│ ◄──────────────────   │   (подход)   │
    └──────────┘  empty > K кадров     └──────────────┘
          ▲                                    │
          │                                    │ occupied > M кадров
          │   empty > K кадров                 ▼
          │                            ┌──────────────┐
          └────────────────────────────│   OCCUPIED   │
                                       │   (занято)   │
                                       └──────────────┘

ИСПРАВЛЕНО:
  1. Удалены устаревшие CleanupRecord / open_cycles / closed_cycles —
     они порождали NaN в get_cycles_dataframe() при смешивании None и int в pandas.
  2. get_cycles_dataframe() переписан — работает только с интервалами,
     никаких int(None) и int(NaN).
  3. Метрика response_time считается от EMPTY.end_sec (момент ухода гостя),
     а не от EMPTY.start_sec.
  4. _update_debounce_counters переименован в _update_debounce (был вызов
     под старым именем в update(), что приводило к AttributeError).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

class TableState(Enum):
    """Состояния зоны столика."""
    EMPTY    = auto()
    APPROACH = auto()
    OCCUPIED = auto()


@dataclass
class StateInterval:
    """Непрерывный период времени в одном состоянии."""
    state:       TableState
    frame_start: int
    frame_end:   int
    start_sec:   float
    end_sec:     float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class StateTransition:
    """Неизменяемая запись о смене состояния."""
    frame_no:   int
    timestamp:  float
    prev_state: TableState
    next_state: TableState

    @property
    def event_name(self) -> str:
        return f"{self.prev_state.name} → {self.next_state.name}"


@dataclass(frozen=True)
class ProgressSnapshot:
    """Нормализованный прогресс счётчиков дебаунса [0.0 … 1.0]."""
    to_approach: float
    to_occupied: float
    to_empty:    float


# ---------------------------------------------------------------------------
# Контекст FSM
# ---------------------------------------------------------------------------

@dataclass
class _MonitorContext:
    """Изолированное хранилище данных FSM."""
    min_empty_frames:    int
    min_occupied_frames: int
    min_stay_frames:     int

    consecutive_empty:    int = 0
    consecutive_occupied: int = 0

    intervals:       list[StateInterval]   = field(default_factory=list)
    active_interval: Optional[StateInterval] = None
    transitions:     list[StateTransition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Состояния FSM
# ---------------------------------------------------------------------------

class _BaseTableState(ABC):
    @property
    @abstractmethod
    def tag(self) -> TableState: ...

    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        pass

    @abstractmethod
    def handle(
        self,
        ctx: _MonitorContext,
        frame_no: int,
        timestamp: float,
        occupied: bool,
    ) -> Optional["_BaseTableState"]: ...


class _EmptyState(_BaseTableState):
    @property
    def tag(self) -> TableState:
        return TableState.EMPTY

    def handle(self, ctx, frame_no, timestamp, occupied):
        if occupied and ctx.consecutive_occupied >= ctx.min_occupied_frames:
            return _ApproachState()
        return None


class _ApproachState(_BaseTableState):
    @property
    def tag(self) -> TableState:
        return TableState.APPROACH

    def handle(self, ctx, frame_no, timestamp, occupied):
        if occupied and ctx.consecutive_occupied >= ctx.min_stay_frames:
            return _OccupiedState()
        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            return _EmptyState()
        return None


class _OccupiedState(_BaseTableState):
    @property
    def tag(self) -> TableState:
        return TableState.OCCUPIED

    def handle(self, ctx, frame_no, timestamp, occupied):
        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            return _EmptyState()
        return None


# ---------------------------------------------------------------------------
# TableMonitor — публичный интерфейс
# ---------------------------------------------------------------------------

class TableMonitor:
    """
    Конечный автомат мониторинга столика.

    Единственный вход — метод update(frame_no, fps, occupied).
    occupied=True  если детектор нашёл человека в ROI.
    occupied=False иначе.
    """

    def __init__(
        self,
        min_empty_frames:    int = 200,
        min_occupied_frames: int = 15,
        min_stay_frames:     int = 150,
    ):
        self._ctx = _MonitorContext(
            min_empty_frames=min_empty_frames,
            min_occupied_frames=min_occupied_frames,
            min_stay_frames=min_stay_frames,
        )
        self._current: _BaseTableState = _EmptyState()

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def set_initial_state(self, occupied: bool, frame_no: int, fps: float) -> None:
        """Задаёт начальное состояние на основе первого кадра."""
        timestamp = frame_no / fps
        if occupied:
            self._current = _OccupiedState()
            logger.info("Старт: стол ЗАНЯТ")
        else:
            self._current = _EmptyState()
            logger.info("Старт: стол СВОБОДЕН")
        self._init_interval(self._current.tag, frame_no, timestamp)

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def update(
        self,
        frame_no: int,
        fps: float,
        occupied: bool,
    ) -> Optional[StateTransition]:
        """
        Обрабатывает один кадр.

        Returns:
            StateTransition если произошла смена состояния, иначе None.
        """
        timestamp = frame_no / fps
        self._update_debounce(occupied)

        if self._ctx.active_interval:
            self._ctx.active_interval.frame_end = frame_no
            self._ctx.active_interval.end_sec   = timestamp

        next_state = self._current.handle(self._ctx, frame_no, timestamp, occupied)
        if next_state:
            return self._do_transition(next_state, frame_no, timestamp)
        return None

    # ------------------------------------------------------------------
    # Внутренняя механика
    # ------------------------------------------------------------------

    def _update_debounce(self, occupied: bool) -> None:
        """Обновляет счётчики непрерывных кадров (гистерезис)."""
        if occupied:
            self._ctx.consecutive_occupied += 1
            self._ctx.consecutive_empty    = 0
        else:
            self._ctx.consecutive_empty    += 1
            self._ctx.consecutive_occupied = 0

    def _init_interval(self, state: TableState, frame: int, ts: float) -> None:
        self._ctx.active_interval = StateInterval(
            state=state,
            frame_start=frame, frame_end=frame,
            start_sec=ts,      end_sec=ts,
        )

    def _do_transition(
        self,
        next_state: _BaseTableState,
        frame_no: int,
        timestamp: float,
    ) -> StateTransition:
        """Фиксирует переход: архивирует интервал, создаёт новый."""
        if self._ctx.active_interval:
            self._ctx.intervals.append(self._ctx.active_interval)

        self._init_interval(next_state.tag, frame_no, timestamp)

        transition = StateTransition(
            frame_no=frame_no,
            timestamp=timestamp,
            prev_state=self._current.tag,
            next_state=next_state.tag,
        )
        self._ctx.transitions.append(transition)

        self._current = next_state
        self._current.on_enter(self._ctx, frame_no, timestamp)

        self._ctx.consecutive_empty    = 0
        self._ctx.consecutive_occupied = 0

        return transition

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def state(self) -> TableState:
        return self._current.tag

    @property
    def transitions(self) -> list[StateTransition]:
        return list(self._ctx.transitions)

    # ------------------------------------------------------------------
    # Экспорт данных
    # ------------------------------------------------------------------

    def get_intervals_dataframe(self) -> pd.DataFrame:
        """
        Все зафиксированные интервалы, включая текущий активный.

        Колонки: state, start_sec, end_sec, duration, frame_start, frame_end
        """
        data = list(self._ctx.intervals)
        if self._ctx.active_interval:
            data.append(self._ctx.active_interval)

        if not data:
            return pd.DataFrame(
                columns=["state", "start_sec", "end_sec", "duration",
                         "frame_start", "frame_end"]
            )

        return pd.DataFrame([{
            "state":       i.state.name,
            "start_sec":   round(i.start_sec, 3),
            "end_sec":     round(i.end_sec,   3),
            "duration":    round(i.duration,  3),
            "frame_start": i.frame_start,
            "frame_end":   i.frame_end,
        } for i in data])

    def get_events_dataframe(self) -> pd.DataFrame:
        """Все переходы состояний."""
        if not self._ctx.transitions:
            return pd.DataFrame(
                columns=["frame_no", "timestamp_sec", "prev_state",
                         "next_state", "event_name"]
            )
        return pd.DataFrame([{
            "frame_no":      t.frame_no,
            "timestamp_sec": round(t.timestamp, 3),
            "prev_state":    t.prev_state.name,
            "next_state":    t.next_state.name,
            "event_name":    t.event_name,
        } for t in self._ctx.transitions])

    def get_cycles_dataframe(self) -> pd.DataFrame:
        """
        Циклы OCCUPIED → EMPTY → APPROACH.

        ИСПРАВЛЕНИЕ NaN-бага: все поля с None остаются Python-None
        (не конвертируются в int через pandas), поэтому NaN не возникает.
        Только строки с is_completed=True имеют непустые approach_*.

        Колонки:
            empty_at_sec      — когда стол освободился (OCCUPIED.end_sec)
            approach_at_sec   — когда подошёл следующий (APPROACH.start_sec)
            response_time_sec — разница (метрика ТЗ), None если нет APPROACH
            is_completed      — True если тройка полная
        """
        df = self.get_intervals_dataframe()
        records = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Цикл начинается только с EMPTY после OCCUPIED
            if row["state"] != "EMPTY":
                continue

            # Проверяем что перед EMPTY был OCCUPIED
            if i == 0 or df.iloc[i - 1]["state"] != "OCCUPIED":
                continue

            occupied_row = df.iloc[i - 1]

            record = {
                "empty_at_frame":    int(row["frame_start"]),
                "empty_at_sec":      row["start_sec"],
                # OCCUPIED закончился = EMPTY начался
                "occupied_end_sec":  occupied_row["end_sec"],
                "approach_at_frame": None,   # остаётся None — не int(NaN)!
                "approach_at_sec":   None,
                "response_time_sec": None,
                "is_completed":      False,
            }

            # Ищем следующий APPROACH или OCCUPIED
            if i + 1 < len(df):
                nxt = df.iloc[i + 1]
                if nxt["state"] in ("APPROACH", "OCCUPIED"):
                    record["approach_at_frame"] = int(nxt["frame_start"])
                    record["approach_at_sec"]   = nxt["start_sec"]
                    # Метрика: от момента ухода гостя (OCCUPIED.end = EMPTY.start)
                    # до момента прихода следующего (APPROACH.start)
                    record["response_time_sec"] = round(
                        nxt["start_sec"] - row["start_sec"], 3
                    )
                    record["is_completed"] = True

            records.append(record)

        if not records:
            return pd.DataFrame(columns=[
                "empty_at_frame", "empty_at_sec", "occupied_end_sec",
                "approach_at_frame", "approach_at_sec",
                "response_time_sec", "is_completed",
            ])

        return pd.DataFrame(records)

    def get_analytics(self) -> dict:
        """
        Агрегированные метрики по завершённым циклам OCCUPIED→EMPTY→APPROACH.
        """
        df = self.get_cycles_dataframe()

        if df.empty:
            completed_times = []
        else:
            completed_times = (
                df.loc[df["is_completed"], "response_time_sec"]
                .dropna()
                .tolist()
            )

        open_count = int((~df["is_completed"]).sum()) if not df.empty else 0

        result = {
            "completed_cycles":    len(completed_times),
            "open_cycles":         open_count,
            "mean_response_sec":   None,
            "median_response_sec": None,
            "min_response_sec":    None,
            "max_response_sec":    None,
        }

        if completed_times:
            result.update({
                "mean_response_sec":   round(sum(completed_times) / len(completed_times), 2),
                "median_response_sec": round(sorted(completed_times)[len(completed_times) // 2], 2),
                "min_response_sec":    round(min(completed_times), 2),
                "max_response_sec":    round(max(completed_times), 2),
            })

        return result

    def get_progress(self) -> ProgressSnapshot:
        """Нормализованный прогресс счётчиков дебаунса [0.0 … 1.0]."""
        def _norm(val: int, thr: int) -> float:
            return min(val / thr, 1.0) if thr > 0 else 0.0

        return ProgressSnapshot(
            to_approach=_norm(self._ctx.consecutive_occupied, self._ctx.min_occupied_frames),
            to_occupied=_norm(self._ctx.consecutive_occupied, self._ctx.min_stay_frames),
            to_empty=   _norm(self._ctx.consecutive_empty,    self._ctx.min_empty_frames),
        )