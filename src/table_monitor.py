"""
Бизнес-логика мониторинга состояния столика (Production Grade)
============================================================

Архитектура: Конечный автомат (FSM) / Паттерн «Состояние»
--------------------------------------------------------
Этот модуль реализует надежную систему мониторинга занятости столиков, предназначенную
для высоконагруженной видеоаналитики. Используется паттерн «Состояние» для инкапсуляции
бизнес-правил, что гарантирует предсказуемость переходов, логики дебаунса (защиты от дребезга)
и генерации аналитики.

Блок-схема работы (Граф состояний):
-----------------------------------
      [ СТАРТ ]
          |
          v
    +-----------+       человек обнаружен (> N кадров)       +--------------+
    |   EMPTY   | ------------------------------------------>|   APPROACH   |
    | (Свободно)| <----------------------------------------- |   (Подход)   |
    +-----------+       человек ушел (> K кадров)            +--------------+
          ^                                                         |
          |                                                         | задержка (> M кадров)
          |             человек ушел (> K кадров)                   v
          |                                                  +--------------+
          +--------------------------------------------------|   OCCUPIED   |
                                                             |   (Занято)   |
                                                             +--------------+

Основные архитектурные концепции:
1. Интервальный трекинг: В отличие от простых логгеров событий, система отслеживает
   непрерывные «Интервалы состояний», что позволяет проводить точный анализ длительности
   (например, «как долго стол был занят», а не просто «когда он стал занятым»).
2. Изоляция контекста: Все изменяемые данные хранятся в `_MonitorContext`.
   Классы состояний предоставляют только логику, что облегчает тестирование и масштабируемость.
3. Дебаунс и гистерезис: Счетчики предотвращают «мерцание» (частую смену состояний),
   вызванное шумом детектора или кратковременными перекрытиями объекта.
4. Обратная совместимость: Поддерживает устаревшие структуры `CleanupRecord` и `StateTransition`
   для обеспечения работы существующих тестов и плагинов отчетности.

Технические требования:
    - Python 3.9+
    - Pandas (для экспорта в DataFrame)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List
import pandas as pd
import logging

# Настройка логгера модуля
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Типы данных (Value Objects и DTO)
# ---------------------------------------------------------------------------

class TableState(Enum):
    """
    Доступные состояния зоны столика.
    EMPTY: Человек не обнаружен.
    APPROACH: Транзитное состояние; человек появился, но еще не задержался надолго.
    OCCUPIED: Подтвержденная занятость.
    """
    EMPTY    = auto()
    APPROACH = auto()
    OCCUPIED = auto()

@dataclass
class StateInterval:
    """Представляет непрерывный период времени, проведенный в определенном состоянии."""
    state: TableState
    frame_start: int
    frame_end: int
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        """Вычисляет длительность в секундах."""
        return self.end_sec - self.start_sec

@dataclass(frozen=True)
class StateTransition:
    """
    Неизменяемая запись о событии смены состояния.
    Сохранена для обратной совместимости с инструментами отчетности.
    """
    frame_no:   int
    timestamp:  float
    prev_state: TableState
    next_state: TableState

    @property
    def event_name(self) -> str:
        """Человекочитаемое название перехода."""
        return f"{self.prev_state.name} → {self.next_state.name}"

@dataclass(frozen=True)
class ProgressSnapshot:
    """Отображает текущий прогресс заполнения счетчиков дебаунса (от 0.0 до 1.0)."""
    to_approach: float
    to_occupied: float
    to_empty:    float

@dataclass
class CleanupRecord:
    """
    Устаревшая метрика: Измеряет «Время реакции» между моментом освобождения
    стола и приходом следующего гостя.
    """
    empty_at_frame:    int
    empty_at_sec:      float
    approach_at_frame: Optional[int]   = None
    approach_at_sec:   Optional[float] = None

    @property
    def response_time_sec(self) -> Optional[float]:
        """Вычисляет время реакции в секундах, если цикл завершен."""
        if self.approach_at_sec is None: return None
        return self.approach_at_sec - self.empty_at_sec

# ---------------------------------------------------------------------------
# Контекст FSM (Контейнер данных)
# ---------------------------------------------------------------------------

@dataclass
class _MonitorContext:
    """
    Внутренний контейнер общего состояния.
    Отделяет логику состояний от хранения данных.
    """
    min_empty_frames:    int  # Кадров для подтверждения EMPTY
    min_occupied_frames: int  # Кадров для входа в APPROACH
    min_stay_frames:     int  # Кадров для подтверждения OCCUPIED

    consecutive_empty:    int = 0
    consecutive_occupied: int = 0

    # Современное хранилище на основе интервалов
    intervals: list[StateInterval] = field(default_factory=list)
    active_interval: Optional[StateInterval] = None

    # Устаревшее хранилище на основе событий
    transitions:   list[StateTransition] = field(default_factory=list)
    open_cycles:   list[CleanupRecord]   = field(default_factory=list)
    closed_cycles: list[CleanupRecord]   = field(default_factory=list)

# ---------------------------------------------------------------------------
# Интерфейс состояния и реализации
# ---------------------------------------------------------------------------

class _BaseTableState(ABC):
    """Абстрактный базовый класс для всех состояний FSM."""
    @property
    @abstractmethod
    def tag(self) -> TableState: ...
    
    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None: 
        """Хуки для логики, выполняемой при входе в состояние (например, аналитика)."""
        pass
        
    @abstractmethod
    def handle(self, ctx: _MonitorContext, frame_no: int, timestamp: float, occupied: bool) -> Optional["_BaseTableState"]: 
        """Оценивает логику переходов для каждого кадра."""
        ...

class _EmptyState(_BaseTableState):
    """Логика состояния EMPTY: стол свободен."""
    @property
    def tag(self) -> TableState: return TableState.EMPTY
    
    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        if not ctx.open_cycles:
            ctx.open_cycles.append(CleanupRecord(empty_at_frame=frame_no, empty_at_sec=timestamp))
            
    def handle(self, ctx: _MonitorContext, frame_no: int, timestamp: float, occupied: bool) -> Optional[_BaseTableState]:
        if occupied and ctx.consecutive_occupied >= ctx.min_occupied_frames:
            return _ApproachState()
        return None

class _ApproachState(_BaseTableState):
    """Логика состояния APPROACH: человек обнаружен, ожидаем подтверждения пребывания."""
    @property
    def tag(self) -> TableState: return TableState.APPROACH
    
    def handle(self, ctx: _MonitorContext, frame_no: int, timestamp: float, occupied: bool) -> Optional[_BaseTableState]:
        if occupied and ctx.consecutive_occupied >= ctx.min_stay_frames:
            return _OccupiedState()
        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            return _EmptyState()
        return None

class _OccupiedState(_BaseTableState):
    """Логика состояния OCCUPIED: занятость подтверждена."""
    @property
    def tag(self) -> TableState: return TableState.OCCUPIED
    
    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        if ctx.open_cycles:
            cycle = ctx.open_cycles.pop()
            cycle.approach_at_frame = frame_no
            cycle.approach_at_sec = timestamp
            ctx.closed_cycles.append(cycle)
            
    def handle(self, ctx: _MonitorContext, frame_no: int, timestamp: float, occupied: bool) -> Optional[_BaseTableState]:
        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            return _EmptyState()
        return None

# ---------------------------------------------------------------------------
# TableMonitor (Публичный оркестратор)
# ---------------------------------------------------------------------------

class TableMonitor:
    """
    Основная точка входа для мониторинга столика.
    Управляет жизненным циклом FSM и предоставляет методы экспорта данных.
    """
    def __init__(self, min_empty_frames=200, min_occupied_frames=15, min_stay_frames=150):
        self._ctx = _MonitorContext(
            min_empty_frames=min_empty_frames,
            min_occupied_frames=min_occupied_frames,
            min_stay_frames=min_stay_frames,
        )
        self._current: _BaseTableState = _EmptyState()

    def set_initial_state(self, occupied: bool, frame_no: int, fps: float):
        """Принудительно устанавливает начальное состояние на основе первого кадра."""
        timestamp = frame_no / fps
        if occupied:
            self._current = _OccupiedState()
            logger.info("Старт: стол ЗАНЯТ")
        else:
            self._current = _EmptyState()
            self._current.on_enter(self._ctx, frame_no, timestamp)
            logger.info("Старт: стол СВОБОДЕН")
        self._init_interval(self._current.tag, frame_no, timestamp)

    def _init_interval(self, state: TableState, frame: int, ts: float):
        """Инициализирует новый интервал отслеживания."""
        self._ctx.active_interval = StateInterval(
            state=state, frame_start=frame, frame_end=frame,
            start_sec=ts, end_sec=ts
        )

    def update(self, frame_no: int, fps: float, occupied: bool) -> Optional[StateTransition]:
        """Обрабатывает один кадр; возвращает объект перехода, если состояние изменилось."""
        timestamp = frame_no / fps
        self._update_debounce_counters(occupied)

        # Продлеваем длительность текущего интервала
        if self._ctx.active_interval:
            self._ctx.active_interval.frame_end = frame_no
            self._ctx.active_interval.end_sec = timestamp

        next_state = self._current.handle(self._ctx, frame_no, timestamp, occupied)
        if next_state:
            return self._do_transition(next_state, frame_no, timestamp)
        return None

    def _do_transition(self, next_state: _BaseTableState, frame_no: int, timestamp: float) -> StateTransition:
        """Выполняет логику перехода и архивацию данных."""
        prev_tag = self._current.tag
        
        if self._ctx.active_interval:
            self._ctx.intervals.append(self._ctx.active_interval)
        
        self._init_interval(next_state.tag, frame_no, timestamp)
        
        transition = StateTransition(
            frame_no=frame_no, timestamp=timestamp,
            prev_state=prev_tag, next_state=next_state.tag
        )
        self._ctx.transitions.append(transition)
        
        self._current = next_state
        self._current.on_enter(self._ctx, frame_no, timestamp)
        
        self._ctx.consecutive_empty = 0
        self._ctx.consecutive_occupied = 0
        return transition

    # --- API аналитики и экспорта данных ---

    def get_intervals_dataframe(self) -> pd.DataFrame:
        """Экспортирует полную временную шкалу интервалов состояний в виде DataFrame."""
        data = self._ctx.intervals.copy()
        if self._ctx.active_interval:
            data.append(self._ctx.active_interval)
            
        return pd.DataFrame([{
            "state": i.state.name,
            "start_sec": round(i.start_sec, 2),
            "end_sec": round(i.end_sec, 2),
            "duration": round(i.duration, 2),
            "frame_start": i.frame_start,
            "frame_end": i.frame_end
        } for i in data])

    def get_analytics(self) -> dict:
        """Вычисляет ключевые показатели эффективности (KPI) времени реакции столика."""
        df = self.get_intervals_dataframe()
        response_times = []
        
        if not df.empty:
            # Ищем пары: интервал EMPTY -> следующий интервал (APPROACH или OCCUPIED)
            for i in range(len(df) - 1):
                curr = df.iloc[i]
                nxt = df.iloc[i+1]
                
                if curr['state'] == TableState.EMPTY.name and nxt['state'] in [TableState.APPROACH.name, TableState.OCCUPIED.name]:
                    # Время реакции: разница между началом нового присутствия и началом периода пустоты
                    delay = nxt['start_sec'] - curr['start_sec']
                    response_times.append(delay)

        res = {
            "total_cycles": len(response_times) + (1 if self.state == TableState.EMPTY else 0),
            "completed_cycles": len(response_times),
            "open_cycles": 1 if self.state == TableState.EMPTY else 0,
            "mean_response_sec": None,
            "median_response_sec": None,
            "min_response_sec": None,
            "max_response_sec": None,
        }
        
        if response_times:
            res.update({
                "mean_response_sec": round(sum(response_times) / len(response_times), 2),
                "median_response_sec": round(sorted(response_times)[len(response_times) // 2], 2),
                "min_response_sec": round(min(response_times), 2),
                "max_response_sec": round(max(response_times), 2),
            })
        return res

    def get_cycles_dataframe(self) -> pd.DataFrame:
        """Экспортирует данные об устаревших циклах уборки для отчетов."""
        df_int = self.get_intervals_dataframe()
        records = []
        
        for i in range(len(df_int)):
            curr = df_int.iloc[i]
            
            if curr['state'] == TableState.EMPTY.name:
                record = {
                    "empty_at_frame": int(curr['frame_start']),
                    "empty_at_sec": round(curr['start_sec'], 3),
                    "approach_at_frame": None,
                    "approach_at_sec": None,
                    "response_time_sec": None,
                    "is_completed": False
                }
                
                # Если за пустотой идет активность — цикл завершен
                if i + 1 < len(df_int):
                    nxt = df_int.iloc[i+1]
                    if nxt['state'] in [TableState.APPROACH.name, TableState.OCCUPIED.name]:
                        record.update({
                            "approach_at_frame": int(nxt['frame_start']),
                            "approach_at_sec": round(nxt['start_sec'], 3),
                            "response_time_sec": round(nxt['start_sec'] - curr['start_sec'], 2),
                            "is_completed": True
                        })
                records.append(record)

        if not records:
            return pd.DataFrame(columns=["empty_at_frame", "empty_at_sec", "approach_at_frame", 
                                         "approach_at_sec", "response_time_sec", "is_completed"])
        
        return pd.DataFrame(records)

    def get_events_dataframe(self) -> pd.DataFrame:
        """Экспортирует моментальные события переходов состояний."""
        if not self._ctx.transitions:
            return pd.DataFrame(columns=["frame_no", "timestamp_sec", "prev_state", "next_state", "event_name"])
        return pd.DataFrame([{
            "frame_no": t.frame_no,
            "timestamp_sec": round(t.timestamp, 3),
            "prev_state": t.prev_state.name,
            "next_state": t.next_state.name,
            "event_name": t.event_name,
        } for t in self._ctx.transitions])

    def get_progress(self) -> ProgressSnapshot:
        """Вычисляет текущий процент удовлетворения пороговых значений дебаунса."""
        def calc(val, thr): return min(val / thr, 1.0) if thr > 0 else 0.0
        return ProgressSnapshot(
            to_approach=calc(self._ctx.consecutive_occupied, self._ctx.min_occupied_frames),
            to_occupied=calc(self._ctx.consecutive_occupied, self._ctx.min_stay_frames),
            to_empty=calc(self._ctx.consecutive_empty, self._ctx.min_empty_frames)
        )

    @property
    def state(self) -> TableState: 
        """Текущее состояние FSM."""
        return self._current.tag

    @property
    def transitions(self) -> list[StateTransition]: 
        """История всех переходов."""
        return list(self._ctx.transitions)

    def _update_debounce_counters(self, occupied: bool) -> None:
        """Внутренняя логика обновления счетчиков, используемых для гистерезиса."""
        if occupied:
            self._ctx.consecutive_occupied += 1
            self._ctx.consecutive_empty = 0
        else:
            self._ctx.consecutive_empty += 1
            self._ctx.consecutive_occupied = 0