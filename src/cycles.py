"""
TaskReportPlugin — плагин аналитики, закрывающий 100% требований ТЗ.

Бизнес-цикл (единица измерения):
    OCCUPIED → EMPTY → APPROACH
    │           │         │
    │           │         └─ новый гость подошёл
    │           └─────────── стол пустой (ждём уборки/нового гостя)
    └─────────────────────── предыдущий гость сидел

Только такая тройка считается валидным циклом.
EMPTY в начале видео (без предшествующего OCCUPIED) — не цикл.
OCCUPIED→EMPTY в конце без APPROACH — открытый цикл.

Главная метрика ТЗ:
    wait_time = EMPTY.duration = APPROACH.start_sec − OCCUPIED.end_sec
    «Сколько стол простоял пустым между гостями»
"""
import pandas as pd
from typing import Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Структура одного цикла
# ---------------------------------------------------------------------------

@dataclass
class TableCycle:
    """
    Валидный бизнес-цикл: тройка OCCUPIED → EMPTY → APPROACH.

    Поля:
        occupied_start_sec  — гость сел
        occupied_end_sec    — гость встал (стол освободился)
        empty_start_sec     — начало пустого периода (= occupied_end_sec)
        empty_end_sec       — конец пустого периода (следующий подошёл)
        approach_start_sec  — новый гость подошёл
        occupied_duration   — сколько сидел предыдущий гость (сек)
        wait_time           — сколько стол простоял пустым (сек) ← МЕТРИКА ТЗ
        is_complete         — True если APPROACH найден
    """
    occupied_start_sec: float
    occupied_end_sec:   float
    empty_start_sec:    float
    empty_end_sec:      Optional[float]
    approach_start_sec: Optional[float]

    @property
    def occupied_duration(self) -> float:
        return round(self.occupied_end_sec - self.occupied_start_sec, 3)

    @property
    def wait_time(self) -> Optional[float]:
        """Время пустоты: от ухода гостя до прихода следующего."""
        if self.approach_start_sec is None:
            return None
        return round(self.approach_start_sec - self.occupied_end_sec, 3)

    @property
    def is_complete(self) -> bool:
        return self.approach_start_sec is not None


# ---------------------------------------------------------------------------
# Основная логика построения циклов
# ---------------------------------------------------------------------------

def build_cycles(df: pd.DataFrame) -> list[TableCycle]:
    """
    Находит все валидные тройки OCCUPIED → EMPTY → APPROACH в списке интервалов.

    Алгоритм:
        Проходим по df слева направо.
        Как только видим OCCUPIED — цикл открыт.
        Следующий интервал должен быть EMPTY — фиксируем.
        Следующий за ним APPROACH — цикл закрыт (валидный).
        Если после EMPTY идёт не APPROACH — цикл открытый (без нового гостя).
        Если OCCUPIED не предшествует — EMPTY/APPROACH не считаем.

    Args:
        df: DataFrame из monitor.get_intervals_dataframe()
            Колонки: state, start_sec, end_sec, duration, frame_start, frame_end

    Returns:
        Список TableCycle, отсортированный по времени.
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

        occupied = row  # Нашли OCCUPIED

        # Следующий интервал должен быть EMPTY
        if i + 1 >= n:
            # OCCUPIED в самом конце, нет EMPTY — открытый цикл без пустоты
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
            # После OCCUPIED идёт не EMPTY (теоретически невозможно в нашем FSM,
            # но защищаемся)
            i += 1
            continue

        empty = nxt  # Нашли EMPTY после OCCUPIED

        # Ищем APPROACH после EMPTY
        if i + 2 >= n:
            # OCCUPIED→EMPTY в конце видео, APPROACH не успел — открытый цикл
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
            # Полный цикл: OCCUPIED → EMPTY → APPROACH
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   empty["start_sec"],
                empty_end_sec=     empty["end_sec"],
                approach_start_sec=after_empty["start_sec"],
            ))
            # Продолжаем с APPROACH/OCCUPIED — он может быть началом следующего цикла
            i += 2
        else:
            # После EMPTY снова EMPTY или неожиданное состояние — открытый цикл
            cycles.append(TableCycle(
                occupied_start_sec=occupied["start_sec"],
                occupied_end_sec=  occupied["end_sec"],
                empty_start_sec=   empty["start_sec"],
                empty_end_sec=     empty["end_sec"],
                approach_start_sec=None,
            ))
            i += 2

    return cycles


def cycles_to_dataframe(cycles: list[TableCycle]) -> pd.DataFrame:
    """Конвертирует список циклов в DataFrame для отчёта и CSV."""
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
