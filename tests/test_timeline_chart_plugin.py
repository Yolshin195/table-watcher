"""
Тест TimelineChartPlugin с настоящими классами TableMonitor и TableState.

Вместо моков — реальный FSM, который прогоняется через последовательности
occupied/not-occupied кадров, точно так же как это делает VideoProcessor.

Запуск:
    python test_timeline_chart.py

Результат: три PNG-файла в папке test_output/
"""

import sys
import logging
from pathlib import Path
import pytest

# --- Добавляем корень проекта в sys.path ---
sys.path.insert(0, str(Path(__file__).parent))

from src.table_monitor import TableMonitor, TableState
from src.plugins import TimelineChartPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

OUT_DIR = Path("test_output")
FPS     = 30.0


# ---------------------------------------------------------------------------
# Вспомогательная функция: строит монитор, скармливая ему кадры
# ---------------------------------------------------------------------------

def build_monitor_from_script(
    script: list[tuple[float, float, bool]],
    total_sec: float,
    fps: float = FPS,
    min_empty_frames:    int = 20,
    min_occupied_frames: int = 5,
    min_stay_frames:     int = 10,
) -> tuple[TableMonitor, int]:
    """
    Прогоняет TableMonitor через синтетические кадры.

    script — список отрезков: (t_start_sec, t_end_sec, occupied)
    Возвращает (monitor, total_frames).
    """
    total_frames = int(total_sec * fps)

    monitor = TableMonitor(
        min_empty_frames=min_empty_frames,
        min_occupied_frames=min_occupied_frames,
        min_stay_frames=min_stay_frames,
    )

    # Определяем начальное состояние по первому отрезку скрипта
    first_occupied = script[0][2] if script else False
    monitor.set_initial_state(occupied=first_occupied, frame_no=0, fps=fps)

    # Построим карту: frame_no → occupied
    # (для каждого кадра ищем подходящий отрезок из скрипта)
    def is_occupied_at(frame: int) -> bool:
        t = frame / fps
        for t_start, t_end, occ in script:
            if t_start <= t < t_end:
                return occ
        return False

    for frame_no in range(total_frames):
        monitor.update(frame_no, fps, is_occupied_at(frame_no))

    return monitor, total_frames


def run_plugin(monitor: TableMonitor, total_frames: int, fps: float,
               filename: str, label: str) -> None:
    """Запускает плагин и сохраняет PNG."""
    plugin = TimelineChartPlugin(output_path=OUT_DIR, filename=filename)
    plugin.on_start(total_frames=total_frames, fps=fps, roi=(100, 80, 300, 200))
    plugin.on_finish(monitor)

    df        = monitor.get_events_dataframe()
    analytics = monitor.get_analytics()

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Переходов зафиксировано : {len(df)}")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"    t={row['timestamp_sec']:>7.1f}s   {row['event_name']}")
    print(f"  Завершённых циклов      : {analytics['completed_cycles']}")
    mean = analytics['mean_response_sec']
    print(f"  Среднее время реакции   : {f'{mean:.1f}с' if mean else 'n/a'}")
    print(f"  График сохранён         : {OUT_DIR / filename}")


# ---------------------------------------------------------------------------
# Сценарий 1: реалистичный — близко к оригинальному скриншоту
#
#   0–30с    OCCUPIED  (стол занят с самого начала видео)
#   30–75с   EMPTY
#   75–82с   APPROACH → EMPTY  (прохожий, не сел)
#   82–530с  EMPTY
#   530–700с OCCUPIED  (гость сел и видео заканчивается)
# ---------------------------------------------------------------------------

@pytest.mark.skip("Тест толко для ручного запуска")
def test_realistic():
    script = [
        (0,   30,  True),   # занят
        (30,  75,  False),  # пусто
        (75,  82,  True),   # прохожий (меньше min_stay_frames)
        (82,  530, False),  # пусто
        (530, 700, True),   # гость
    ]
    monitor, total_frames = build_monitor_from_script(script, total_sec=700)
    run_plugin(monitor, total_frames, FPS,
               "scenario_realistic.png",
               "Реалистичный: занят → пусто → прохожий → пусто → гость")


# ---------------------------------------------------------------------------
# Сценарий 2: стол изначально пустой, два полных цикла
#
#   0–60с    EMPTY
#   60–200с  OCCUPIED
#   200–260с EMPTY
#   260–400с OCCUPIED
#   400–500с EMPTY  (видео заканчивается пока пусто)
# ---------------------------------------------------------------------------

@pytest.mark.skip("Тест толко для ручного запуска")
def test_starts_empty_two_cycles():
    script = [
        (0,   60,  False),
        (60,  200, True),
        (200, 260, False),
        (260, 400, True),
        (400, 500, False),
    ]
    monitor, total_frames = build_monitor_from_script(script, total_sec=500)
    run_plugin(monitor, total_frames, FPS,
               "scenario_two_cycles.png",
               "Два полных цикла: стол начинает пустым")


# ---------------------------------------------------------------------------
# Сценарий 3: много коротких появлений (прохожие) + один настоящий гость
#
#   0–100с   EMPTY
#   несколько «прохожих» по 3–4с
#   300–600с OCCUPIED (настоящий гость)
#   600–700с EMPTY
# ---------------------------------------------------------------------------

@pytest.mark.skip("Тест толко для ручного запуска")
def test_many_passers():
    script = [
        (0,    100,  False),
        (100,  103,  True),   # прохожий 1
        (103,  150,  False),
        (150,  154,  True),   # прохожий 2
        (154,  200,  False),
        (200,  203,  True),   # прохожий 3
        (203,  300,  False),
        (300,  600,  True),   # настоящий гость
        (600,  700,  False),
    ]
    monitor, total_frames = build_monitor_from_script(script, total_sec=700)
    run_plugin(monitor, total_frames, FPS,
               "scenario_many_passers.png",
               "Много прохожих + один настоящий гость")


# ---------------------------------------------------------------------------
# Запуск всех сценариев
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    print("🚀 Запуск тестов TimelineChartPlugin (настоящие классы)\n")

    test_realistic()
    test_starts_empty_two_cycles()
    test_many_passers()

    print(f"\n✅ Все тесты завершены. Графики в папке: {OUT_DIR.resolve()}/")