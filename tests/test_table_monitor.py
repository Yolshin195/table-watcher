"""
Бизнес-логика мониторинга состояния столика.

Архитектура: Паттерн «Машина состояний» (State Machine / FSM).
Каждое состояние — отдельный класс, инкапсулирующий:
  - бизнес-правило: «когда переходить в другое состояние»
  - аналитику: «что записать при входе в это состояние»

Достаточно посмотреть на класс состояния — и сразу понятно его поведение.

Граф переходов:
                   человек появился (>= min_occupied_frames)
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              ▼
  EMPTY                                                        APPROACH
    ▲                              задержался (>= min_stay_frames) │
    │  прошёл мимо (>= min_empty_frames) ◄─────────────────────────┘
    │
    │         ушёл (>= min_empty_frames)
  EMPTY  ◄──────────────────────────────────────────────────  OCCUPIED
    ▲                                                              │
    └──────────────────────────────────────────────────────────────┘

Соответствие ТЗ:
  ✅ Три состояния: EMPTY, OCCUPIED, APPROACH
  ✅ Дебаунс (защита от мерцания детектора)
  ✅ Временные метки в Pandas DataFrame
  ✅ Аналитика: среднее время между уходом гостя и подходом следующего
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Типы данных (Value Objects)
# ---------------------------------------------------------------------------

class TableState(Enum):
    """
    Состояния столика.

    EMPTY    — в зоне ROI никого нет, стол свободен.
    APPROACH — кто-то появился после периода пустоты; ждём подтверждения:
               задержался → OCCUPIED, ушёл сразу → EMPTY (прошёл мимо).
    OCCUPIED — человек подтверждённо находится за столом.
    """
    EMPTY    = auto()
    APPROACH = auto()
    OCCUPIED = auto()


@dataclass(frozen=True)
class StateTransition:
    """Иммутабельный факт смены состояния."""
    frame_no:   int
    timestamp:  float        # секунды от начала видео
    prev_state: TableState
    next_state: TableState

    @property
    def event_name(self) -> str:
        return f"{self.prev_state.name} → {self.next_state.name}"


@dataclass
class CleanupRecord:
    """
    Одна запись о цикле «стол освободился → подтверждённый гость пришёл».

    approach_at_* заполняется только когда APPROACH подтвердился в OCCUPIED.
    Если человек прошёл мимо (APPROACH → EMPTY), цикл остаётся открытым.
    """
    empty_at_frame:    int
    empty_at_sec:      float
    approach_at_frame: Optional[int]   = None
    approach_at_sec:   Optional[float] = None

    @property
    def response_time_sec(self) -> Optional[float]:
        """Время реакции: сколько секунд стол ждал между гостями."""
        if self.approach_at_sec is None:
            return None
        return self.approach_at_sec - self.empty_at_sec


# ---------------------------------------------------------------------------
# Контекст — хранилище данных, доступных всем состояниям
# ---------------------------------------------------------------------------

@dataclass
class _MonitorContext:
    """
    Разделяемый контекст FSM.

    Состояния читают и пишут сюда — единственный канал обмена данными.
    Сам класс не содержит никакой логики, только данные.
    """
    min_empty_frames:    int   # кадров пустоты для перехода в EMPTY
    min_occupied_frames: int   # кадров присутствия для входа в APPROACH
    min_stay_frames:     int   # кадров в APPROACH для подтверждения OCCUPIED

    consecutive_empty:    int = 0
    consecutive_occupied: int = 0

    transitions:   list[StateTransition] = field(default_factory=list)
    open_cycles:   list[CleanupRecord]   = field(default_factory=list)
    closed_cycles: list[CleanupRecord]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Базовый класс состояния (принцип Open/Closed из SOLID)
# ---------------------------------------------------------------------------

class _BaseTableState(ABC):
    """
    Абстрактный класс состояния столика.

    Каждый наследник отвечает ровно за одно состояние и реализует:
      tag        — идентификатор состояния (значение TableState)
      on_enter() — побочные эффекты при входе в состояние (аналитика)
      handle()   — бизнес-правило: в какое состояние перейти на этом кадре?

    Принцип Single Responsibility: один класс = одно состояние = одно правило.
    """

    @property
    @abstractmethod
    def tag(self) -> TableState:
        """Идентификатор этого состояния."""

    @abstractmethod
    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        """Вызывается один раз при входе. Здесь — аналитика и побочные эффекты."""

    @abstractmethod
    def handle(
        self,
        ctx: _MonitorContext,
        frame_no: int,
        timestamp: float,
        occupied: bool,
    ) -> Optional["_BaseTableState"]:
        """
        Проверить условие перехода на текущем кадре.

        Returns:
            Следующее состояние — если пора переходить.
            None              — если остаёмся в текущем.
        """


# ---------------------------------------------------------------------------
# Конкретные состояния — здесь живут ВСЕ бизнес-правила системы
# ---------------------------------------------------------------------------

class _EmptyState(_BaseTableState):
    """
    Состояние: СТОЛ ПУСТОЙ.

    Бизнес-правило перехода:
      Если человек появляется в зоне ROI и удерживается не менее
      min_occupied_frames кадров подряд → переходим в APPROACH.

    Аналитика при входе:
      Открываем новый цикл ожидания — фиксируем момент, когда стол стал пустым.
      (ТЗ: «Для каждого случая, когда стол стал пустым, определи,
             через какое время к нему подошёл следующий человек»)
    """

    @property
    def tag(self) -> TableState:
        return TableState.EMPTY

    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        ctx.open_cycles.append(CleanupRecord(
            empty_at_frame=frame_no,
            empty_at_sec=timestamp,
        ))

    def handle(
        self,
        ctx: _MonitorContext,
        frame_no: int,
        timestamp: float,
        occupied: bool,
    ) -> Optional[_BaseTableState]:
        if occupied and ctx.consecutive_occupied >= ctx.min_occupied_frames:
            # ТЗ: «Подход к столу (появление человека в зоне после периода пустоты)»
            return _ApproachState()
        return None


class _ApproachState(_BaseTableState):
    """
    Состояние: ПОДХОД К СТОЛУ (период наблюдения).

    Человек появился — но мы ещё не знаем, сел ли он или проходит мимо.
    Остаёмся в этом состоянии, пока не накопится достаточно кадров в ту
    или иную сторону.

    Бизнес-правило перехода:
      Человек задержался (>= min_stay_frames кадров присутствия подряд)
        → OCCUPIED: гость подтверждён, закрываем цикл ожидания.
      Человек исчез (>= min_empty_frames кадров отсутствия подряд)
        → EMPTY: прошёл мимо, цикл ожидания остаётся открытым.

    Аналитика при входе:
      Намеренно ничего не делаем — цикл закроет _OccupiedState,
      только когда присутствие подтверждено. «Прохожий» в статистику
      времени отклика не попадёт.
    """

    @property
    def tag(self) -> TableState:
        return TableState.APPROACH

    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        pass  # ждём подтверждения; аналитику пишет _OccupiedState при входе

    def handle(
        self,
        ctx: _MonitorContext,
        frame_no: int,
        timestamp: float,
        occupied: bool,
    ) -> Optional[_BaseTableState]:
        if occupied and ctx.consecutive_occupied >= ctx.min_stay_frames:
            # Человек задержался достаточно долго → это гость, не прохожий
            return _OccupiedState()

        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            # Человек ушёл быстро → прошёл мимо, стол снова пустой
            return _EmptyState()

        return None  # ещё не ясно — наблюдаем дальше


class _OccupiedState(_BaseTableState):
    """
    Состояние: СТОЛ ЗАНЯТ.

    Бизнес-правило перехода:
      Если зона ROI пустует не менее min_empty_frames кадров подряд → EMPTY.

    Аналитика при входе:
      Закрываем открытый цикл ожидания — фиксируем подтверждённый момент
      появления гостя. Делаем это здесь, а не при входе в APPROACH, чтобы
      «прохожий» не учитывался в статистике времени отклика.
      (ТЗ: «Среднее время между уходом гостя и подходом следующего человека»)
    """

    @property
    def tag(self) -> TableState:
        return TableState.OCCUPIED

    def on_enter(self, ctx: _MonitorContext, frame_no: int, timestamp: float) -> None:
        # Цикл закрываем только сейчас — присутствие гостя подтверждено.
        if ctx.open_cycles:
            cycle = ctx.open_cycles.pop()
            cycle.approach_at_frame = frame_no
            cycle.approach_at_sec   = timestamp
            ctx.closed_cycles.append(cycle)

    def handle(
        self,
        ctx: _MonitorContext,
        frame_no: int,
        timestamp: float,
        occupied: bool,
    ) -> Optional[_BaseTableState]:
        if not occupied and ctx.consecutive_empty >= ctx.min_empty_frames:
            # ТЗ: «Стол пустой (людей в зоне нет)»
            return _EmptyState()
        return None


# ---------------------------------------------------------------------------
# TableMonitor — оркестратор FSM
# ---------------------------------------------------------------------------

class TableMonitor:
    """
    Конечный автомат (FSM) для отслеживания состояния одного столика.

    Принцип работы:
        Вызывай update() на каждом кадре, передавая:
          - frame_no : порядковый номер кадра (0-based)
          - fps      : частота кадров видео (для перевода в секунды)
          - occupied : True если детектор обнаружил человека в зоне ROI

        Класс сам управляет дебаунсом, историей переходов и аналитикой.

    Параметры (все в кадрах):
        min_empty_frames    : сколько кадров пустоты нужно для перехода в EMPTY.
                              Защищает от мерцания детектора.
                              По умолчанию 30 (~1 сек при 30fps).

        min_occupied_frames : сколько кадров присутствия нужно для входа в APPROACH.
                              По умолчанию 5 (~0.17 сек при 30fps).

        min_stay_frames     : сколько кадров нужно пробыть в APPROACH,
                              чтобы перейти в OCCUPIED («гость», а не «прохожий»).
                              Бизнес-смысл: если официант лишь убрал тарелку и ушёл,
                              это не считается «занятием» стола.
                              По умолчанию 15 (~0.5 сек при 30fps).
    """

    def __init__(
        self,
        min_empty_frames:    int = 30,
        min_occupied_frames: int = 5,
        min_stay_frames:     int = 15,
    ):
        self._ctx = _MonitorContext(
            min_empty_frames=min_empty_frames,
            min_occupied_frames=min_occupied_frames,
            min_stay_frames=min_stay_frames,
        )
        # Начальное состояние — стол пустой.
        # on_enter не вызываем: нет предыдущего гостя, цикл ожидания не нужен.
        self._current: _BaseTableState = _EmptyState()

    # -----------------------------------------------------------------------
    # Публичный API
    # -----------------------------------------------------------------------

    @property
    def state(self) -> TableState:
        """Текущее подтверждённое состояние столика."""
        return self._current.tag

    @property
    def transitions(self) -> list[StateTransition]:
        """Копия списка всех зафиксированных переходов состояний."""
        return list(self._ctx.transitions)

    def update(self, frame_no: int, fps: float, occupied: bool) -> Optional[StateTransition]:
        """
        Обработать один кадр.

        Args:
            frame_no : порядковый номер кадра (0-based)
            fps      : частота кадров видео
            occupied : True если в зоне ROI обнаружен человек

        Returns:
            StateTransition если состояние изменилось на этом кадре, иначе None.
        """
        timestamp = frame_no / fps
        self._update_debounce_counters(occupied)

        next_state = self._current.handle(self._ctx, frame_no, timestamp, occupied)
        if next_state is None:
            return None

        return self._do_transition(next_state, frame_no, timestamp)

    def get_analytics(self) -> dict:
        """
        Итоговая аналитика по всем завершённым циклам.

        Завершённый цикл: стол стал пустым → появился подтверждённый гость.
        Незавершённый: видео кончилось пока стол пустой или в состоянии APPROACH.

        Ключевая метрика (ТЗ): среднее время между уходом гостя и подходом следующего.
        """
        times = [
            c.response_time_sec
            for c in self._ctx.closed_cycles
            if c.response_time_sec is not None
        ]

        if not times:
            return {
                "total_cycles":        0,
                "completed_cycles":    0,
                "open_cycles":         len(self._ctx.open_cycles),
                "mean_response_sec":   None,
                "median_response_sec": None,
                "min_response_sec":    None,
                "max_response_sec":    None,
            }

        return {
            "total_cycles":        len(self._ctx.closed_cycles) + len(self._ctx.open_cycles),
            "completed_cycles":    len(self._ctx.closed_cycles),
            "open_cycles":         len(self._ctx.open_cycles),
            "mean_response_sec":   round(sum(times) / len(times), 2),
            "median_response_sec": round(sorted(times)[len(times) // 2], 2),
            "min_response_sec":    round(min(times), 2),
            "max_response_sec":    round(max(times), 2),
        }

    def get_events_dataframe(self) -> pd.DataFrame:
        """
        Все события в виде Pandas DataFrame.
        Колонки: frame_no, timestamp_sec, prev_state, next_state, event_name.
        """
        if not self._ctx.transitions:
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
            for t in self._ctx.transitions
        ])

    def get_cycles_dataframe(self) -> pd.DataFrame:
        """
        Все циклы «стол освободился → подтверждённый гость» в виде DataFrame.
        Колонки: empty_at_sec, approach_at_sec, response_time_sec, is_completed.
        """
        all_cycles = self._ctx.closed_cycles + self._ctx.open_cycles
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
                "is_completed":      c in self._ctx.closed_cycles,
            }
            for c in all_cycles
        ])

    # -----------------------------------------------------------------------
    # Свойства для совместимости (доступ к счётчикам дебаунса)
    # -----------------------------------------------------------------------

    @property
    def _consecutive_occupied(self) -> int:
        """Текущий счётчик непрерывного присутствия. Используется в тестах."""
        return self._ctx.consecutive_occupied

    @property
    def _consecutive_empty(self) -> int:
        """Текущий счётчик непрерывного отсутствия. Используется в тестах."""
        return self._ctx.consecutive_empty

    # -----------------------------------------------------------------------
    # Внутренняя механика FSM
    # -----------------------------------------------------------------------

    def _update_debounce_counters(self, occupied: bool) -> None:
        """
        Обновить счётчики непрерывного присутствия/отсутствия.
        При смене направления противоположный счётчик сбрасывается в ноль.
        """
        if occupied:
            self._ctx.consecutive_occupied += 1
            self._ctx.consecutive_empty    = 0
        else:
            self._ctx.consecutive_empty    += 1
            self._ctx.consecutive_occupied = 0

    def _do_transition(
        self,
        next_state: _BaseTableState,
        frame_no: int,
        timestamp: float,
    ) -> StateTransition:
        """
        Выполнить переход в новое состояние:
          1. Записать переход в историю.
          2. Вызвать on_enter() нового состояния (аналитика).
          3. Установить новое состояние как текущее.
          4. Сбросить счётчики дебаунса — новое состояние начинает считать с нуля.
        """
        transition = StateTransition(
            frame_no=frame_no,
            timestamp=timestamp,
            prev_state=self._current.tag,
            next_state=next_state.tag,
        )
        self._ctx.transitions.append(transition)
        next_state.on_enter(self._ctx, frame_no, timestamp)
        self._current = next_state

        # Сбрасываем счётчики: новое состояние должно накопить свои кадры заново.
        # Это гарантирует, что из APPROACH нельзя выйти мгновенно.
        self._ctx.consecutive_empty    = 0
        self._ctx.consecutive_occupied = 0

        return transition