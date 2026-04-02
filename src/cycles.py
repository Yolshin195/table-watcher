"""
src/analytics/cycles.py
=======================

Бизнес-цикл столика и алгоритм его построения из интервалов FSM.

Единица измерения — тройка:
    OCCUPIED → EMPTY → APPROACH
    │           │         │
    │           │         └─ новый гость подошёл
    │           └─────────── стол пустой (ждём уборки / нового гостя)
    └─────────────────────── предыдущий гость сидел

Только такая тройка считается валидным (завершённым) циклом:
  - EMPTY без предшествующего OCCUPIED — не цикл (начало видео).
  - OCCUPIED → EMPTY без APPROACH — открытый цикл (конец видео или данных мало).

Главная метрика ТЗ:
    wait_time = APPROACH.start_sec − OCCUPIED.end_sec
    «Сколько стол простоял пустым между гостями»

Импорт:
    from src.analytics.cycles import TableCycle, build_cycles, cycles_to_dataframe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Структура одного бизнес-цикла
# ---------------------------------------------------------------------------

@dataclass
class TableCycle:
    """
    Валидный бизнес-цикл: тройка OCCUPIED → EMPTY → APPROACH.

    Поля (все временны́е метки в секундах от начала видео):
        occupied_start_sec  — гость сел (начало OCCUPIED-интервала)
        occupied_end_sec    — гость встал, стол освободился (конец OCCUPIED)
        empty_start_sec     — начало пустого периода (= occupied_end_sec)
        empty_end_sec       — конец пустого периода (= approach_start_sec),
                              None если APPROACH ещё не зафиксирован
        approach_start_sec  — новый гость подошёл (начало APPROACH-интервала),
                              None для открытых циклов

    Вычисляемые свойства:
        occupied_duration   — сколько сидел предыдущий гость (сек)
        wait_time           — сколько стол простоял пустым (сек) ← МЕТРИКА ТЗ
        is_complete         — True если APPROACH найден (цикл закрыт)
    """
    occupied_start_sec: float
    occupied_end_sec:   float
    empty_start_sec:    float
    empty_end_sec:      Optional[float]
    approach_start_sec: Optional[float]

    @property
    def occupied_duration(self) -> float:
        """Сколько гость сидел (сек)."""
        return round(self.occupied_end_sec - self.occupied_start_sec, 3)

    @property
    def wait_time(self) -> Optional[float]:
        """
        Время пустоты стола: от ухода гостя до прихода следующего.
        Возвращает None для открытых циклов (нет APPROACH).
        """
        if self.approach_start_sec is None:
            return None
        return round(self.approach_start_sec - self.occupied_end_sec, 3)

    @property
    def is_complete(self) -> bool:
        """True если тройка OCCUPIED → EMPTY → APPROACH полностью зафиксирована."""
        return self.approach_start_sec is not None


# ---------------------------------------------------------------------------
# Построение циклов из DataFrame интервалов
# ---------------------------------------------------------------------------

def build_cycles(df: pd.DataFrame) -> list[TableCycle]:
    """
    Находит все тройки OCCUPIED → EMPTY → APPROACH в таблице интервалов.

    Алгоритм (однопроходный, O(n)):
        Идём по строкам слева направо.
        OCCUPIED     → открываем цикл, запоминаем строку.
        Следующий    → должен быть EMPTY; если нет — пропускаем OCCUPIED.
        После EMPTY  → если есть APPROACH/OCCUPIED — цикл закрыт (is_complete=True).
                       иначе — цикл открытый (is_complete=False).
        EMPTY без предшествующего OCCUPIED — не является началом цикла.

    Args:
        df: DataFrame из ``monitor.get_intervals_dataframe()``.
            Ожидаемые колонки: state, start_sec, end_sec, duration,
            frame_start, frame_end.

    Returns:
        Список ``TableCycle``, отсортированный по времени (порядок строк df).
    """
    cycles: list[TableCycle] = []
    i = 0
    n = len(df)

    while i < n:
        row = df.iloc[i]

        # Ищем начало цикла — интервал OCCUPIED
        if row["state"] != "OCCUPIED":
            i += 1
            continue

        occupied = row

        # Следующий интервал должен быть EMPTY
        if i + 1 >= n:
            # OCCUPIED в самом конце, нет EMPTY — открытый цикл
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   occupied["end_sec"],
                empty_end_sec=     None,
                approach_start_sec=None,
            ))
            i += 1
            continue

        nxt = df.iloc[i + 1]

        if nxt["state"] != "EMPTY":
            # После OCCUPIED не EMPTY — теоретически невозможно в нашем FSM,
            # но защищаемся от неожиданных данных
            i += 1
            continue

        empty = nxt

        # Ищем APPROACH или OCCUPIED после EMPTY
        if i + 2 >= n:
            # OCCUPIED → EMPTY в конце видео, APPROACH не успел — открытый цикл
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   empty["start_sec"],
                empty_end_sec=     empty["end_sec"],
                approach_start_sec=None,
            ))
            i += 2
            continue

        after_empty = df.iloc[i + 2]

        if after_empty["state"] in ("APPROACH", "OCCUPIED"):
            # Полный (завершённый) цикл
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   empty["start_sec"],
                empty_end_sec=     empty["end_sec"],
                approach_start_sec=after_empty["start_sec"],
            ))
            # after_empty может сам быть началом следующего цикла — не пропускаем
            i += 2
        else:
            # После EMPTY неожиданное состояние — открытый цикл
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   empty["start_sec"],
                empty_end_sec=     empty["end_sec"],
                approach_start_sec=None,
            ))
            i += 2

    return cycles


# ---------------------------------------------------------------------------
# Конвертация в DataFrame
# ---------------------------------------------------------------------------

def cycles_to_dataframe(cycles: list[TableCycle]) -> pd.DataFrame:
    """
    Конвертирует список ``TableCycle`` в ``pd.DataFrame``.

    Колонки:
        occupied_start_sec  — гость сел
        occupied_end_sec    — гость ушёл
        occupied_duration   — сколько сидел (сек)
        empty_start_sec     — начало пустого периода
        empty_end_sec       — конец пустого периода (None если нет APPROACH)
        approach_start_sec  — подошёл новый гость (None для открытых циклов)
        wait_time           — время пустоты (сек) ← главная метрика ТЗ
        is_complete         — True если цикл завершён
    """
    if not cycles:
        return pd.DataFrame(columns=[
            "occupied_start_sec", "occupied_end_sec", "occupied_duration",
            "empty_start_sec", "empty_end_sec",
            "approach_start_sec", "wait_time", "is_complete",
        ])

    return pd.DataFrame([{
        "occupied_start_sec": c.occupied_start_sec,
        "occupied_end_sec":   c.occupied_end_sec,
        "occupied_duration":  c.occupied_duration,
        "empty_start_sec":    c.empty_start_sec,
        "empty_end_sec":      c.empty_end_sec,
        "approach_start_sec": c.approach_start_sec,
        "wait_time":          c.wait_time,      # ← главная метрика ТЗ
        "is_complete":        c.is_complete,
    } for c in cycles])
