"""
Тесты для TableMonitor.

Запуск:
    pytest tests/test_main.py -v

Ключевое изменение по сравнению с исходной версией:
    APPROACH теперь является настоящим состоянием ожидания, а не мгновенным
    событием. После входа в APPROACH нужно ещё min_stay_frames кадров
    присутствия, чтобы перейти в OCCUPIED. Счётчики дебаунса сбрасываются
    при каждом переходе, поэтому кадры в APPROACH «не считаются» для OCCUPIED.

    Тесты, которые раньше ожидали мгновенного OCCUPIED после N кадров,
    обновлены: теперь они явно проводят монитор через APPROACH в OCCUPIED.
"""

import pandas as pd
import pytest
from src.table_monitor import TableMonitor, TableState, StateTransition


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

FPS = 10.0  # 1 кадр = 0.1 сек


def feed(monitor: TableMonitor, occupied: bool, n_frames: int, start_frame: int = 0):
    """Подать N одинаковых кадров подряд. Возвращает список переходов (без None)."""
    transitions = []
    for i in range(n_frames):
        t = monitor.update(start_frame + i, FPS, occupied)
        if t is not None:
            transitions.append(t)
    return transitions


def make_monitor(empty=3, occupied=2, stay=None) -> TableMonitor:
    """
    Монитор с маленькими порогами — удобно для тестов.

    stay=None (по умолчанию) → stay=occupied.
    Это означает: столько же кадров для выхода из APPROACH, сколько для входа.
    Используйте stay=0 для мгновенного перехода APPROACH→OCCUPIED (старое поведение).
    """
    return TableMonitor(
        min_empty_frames=empty,
        min_occupied_frames=occupied,
        min_stay_frames=stay if stay is not None else occupied,
    )


def occupy(m: TableMonitor, n_occupied: int, start_frame: int = 0) -> int:
    """
    Вспомогательная функция: провести монитор через APPROACH в OCCUPIED.

    Подаёт n_occupied кадров (для входа в APPROACH), затем ещё столько же
    (для выхода из APPROACH в OCCUPIED). Возвращает следующий номер кадра.

    Использование:
        next_frame = occupy(m, n_occupied=2, start_frame=0)
    """
    feed(m, True, n_occupied, start_frame)
    feed(m, True, n_occupied, start_frame + n_occupied)
    return start_frame + n_occupied * 2


# ---------------------------------------------------------------------------
# Группа 1: Дебаунс
# ---------------------------------------------------------------------------

class TestDebounce:

    def test_no_transition_before_threshold_empty(self):
        """Стол не переходит в EMPTY раньше чем накопится min_empty_frames."""
        m = make_monitor(empty=5, occupied=2)
        # Сначала занимаем стол (через APPROACH)
        occupy(m, n_occupied=2)
        assert m.state == TableState.OCCUPIED

        # Подаём 4 пустых кадра — один меньше порога
        transitions = feed(m, False, 4, start_frame=10)
        assert m.state == TableState.OCCUPIED
        assert len(transitions) == 0

    def test_transition_exactly_at_threshold_empty(self):
        """Переход в EMPTY происходит ровно на кадре N = min_empty_frames."""
        m = make_monitor(empty=5, occupied=2)
        occupy(m, n_occupied=2)
        transitions = feed(m, False, 5, start_frame=10)
        assert m.state == TableState.EMPTY
        assert len(transitions) == 1
        assert transitions[0].next_state == TableState.EMPTY

    def test_no_transition_before_threshold_occupied(self):
        """Стол не помечается занятым раньше чем накопится min_occupied_frames."""
        m = make_monitor(empty=3, occupied=4)
        # Подаём 3 кадра с человеком — один меньше порога входа в APPROACH
        transitions = feed(m, True, 3)
        assert m.state == TableState.EMPTY
        assert len(transitions) == 0

    def test_transition_exactly_at_threshold_occupied(self):
        """
        Переход в APPROACH происходит ровно на кадре N = min_occupied_frames.
        После APPROACH нужно ещё min_stay_frames кадров для OCCUPIED.
        """
        m = make_monitor(empty=3, occupied=4)
        # Ровно 4 кадра → входим в APPROACH
        approach_ts = feed(m, True, 4)
        assert len(approach_ts) == 1
        assert approach_ts[0].next_state == TableState.APPROACH

        # Ещё 4 кадра (stay=occupied=4) → OCCUPIED
        feed(m, True, 4, start_frame=4)
        assert m.state == TableState.OCCUPIED

    def test_debounce_resets_on_interruption(self):
        """Прерывание до порога обнуляет счётчик — переход не происходит."""
        m = make_monitor(empty=5, occupied=2)
        occupy(m, n_occupied=2)
        assert m.state == TableState.OCCUPIED

        # 4 пустых, потом 1 занятый — счётчик пустоты сбрасывается
        feed(m, False, 4, start_frame=10)
        assert m.state == TableState.OCCUPIED

        feed(m, True, 1, start_frame=14)
        assert m.state == TableState.OCCUPIED

        # ещё 4 пустых — снова не хватает до 5
        transitions = feed(m, False, 4, start_frame=15)
        assert m.state == TableState.OCCUPIED
        assert len(transitions) == 0

    def test_only_one_transition_per_threshold_crossing(self):
        """Переход фиксируется ровно один раз, не повторяется на каждом кадре."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        transitions = feed(m, False, 10, start_frame=10)
        assert len(transitions) == 1
        assert transitions[0].next_state == TableState.EMPTY


# ---------------------------------------------------------------------------
# Группа 2: Переходы FSM
# ---------------------------------------------------------------------------

class TestFSMTransitions:

    def test_initial_state_is_empty(self):
        m = make_monitor()
        assert m.state == TableState.EMPTY

    def test_empty_to_approach_on_first_person(self):
        """Первый человек после пустоты → APPROACH, не OCCUPIED."""
        m = make_monitor()
        t = feed(m, True, 2)[0]
        assert t.prev_state == TableState.EMPTY
        assert t.next_state == TableState.APPROACH

    def test_after_approach_state_is_occupied(self):
        """После APPROACH + min_stay_frames кадров состояние становится OCCUPIED."""
        m = make_monitor()
        feed(m, True, 2)            # → APPROACH
        feed(m, True, 2, start_frame=2)  # → OCCUPIED (stay=2)
        assert m.state == TableState.OCCUPIED

    def test_occupied_to_empty(self):
        """Человек ушёл → стол переходит в EMPTY."""
        m = make_monitor()
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        assert m.state == TableState.EMPTY

    def test_full_cycle(self):
        """Полный цикл: EMPTY → APPROACH → OCCUPIED → EMPTY → APPROACH → OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        frame = 0

        # Первый гость: через APPROACH в OCCUPIED
        frame = occupy(m, n_occupied=2, start_frame=frame)
        assert m.state == TableState.OCCUPIED

        # Гость уходит
        feed(m, False, 3, frame); frame += 3
        assert m.state == TableState.EMPTY

        # Второй гость: снова APPROACH → OCCUPIED
        feed(m, True, 2, frame); frame += 2
        assert m.state == TableState.APPROACH
        feed(m, True, 2, frame)
        assert m.state == TableState.OCCUPIED

    def test_transition_returns_correct_object(self):
        """update() возвращает StateTransition с правильными полями."""
        m = make_monitor(empty=3, occupied=2)
        result = None
        for i in range(2):
            result = m.update(i, FPS, True)
        assert result is not None
        assert isinstance(result, StateTransition)
        assert result.frame_no == 1
        assert abs(result.timestamp - 1 / FPS) < 1e-9

    def test_no_approach_if_already_occupied(self):
        """APPROACH не должен срабатывать если стол и так OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)          # первый раз APPROACH → OCCUPIED
        assert m.state == TableState.OCCUPIED

        # Короткий перерыв — меньше порога дебаунса
        feed(m, False, 2, start_frame=10)
        assert m.state == TableState.OCCUPIED  # не успел переключиться

        # Снова человек — не должно быть APPROACH (стол уже OCCUPIED)
        transitions = feed(m, True, 2, start_frame=12)
        assert not any(t.next_state == TableState.APPROACH for t in transitions)


# ---------------------------------------------------------------------------
# Группа 3: Логика APPROACH
# ---------------------------------------------------------------------------

class TestApproachLogic:

    def test_approach_only_after_confirmed_empty(self):
        """APPROACH срабатывает только после подтверждённого EMPTY, не раньше."""
        m = make_monitor(empty=5, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 4, start_frame=10)   # 4 пустых — не хватает до EMPTY
        transitions = feed(m, True, 2, start_frame=14)
        assert all(t.next_state != TableState.APPROACH for t in transitions)

    def test_approach_closes_open_cycle(self):
        """После APPROACH+OCCUPIED в _closed_cycles появляется запись."""
        m = make_monitor()
        occupy(m, n_occupied=2)              # первый гость
        feed(m, False, 3, start_frame=10)    # уходит → EMPTY (open cycle)
        feed(m, True, 2, start_frame=13)     # второй гость → APPROACH
        feed(m, True, 2, start_frame=15)     # → OCCUPIED (closes cycle)

        analytics = m.get_analytics()
        assert analytics["completed_cycles"] == 1
        assert analytics["open_cycles"] == 0

    def test_approach_transition_recorded_in_history(self):
        """APPROACH и OCCUPIED появляются в списке transitions в правильном порядке."""
        m = make_monitor()
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        states = [t.next_state for t in m.transitions]
        assert TableState.APPROACH in states
        assert TableState.OCCUPIED in states
        # APPROACH должен предшествовать OCCUPIED
        assert states.index(TableState.APPROACH) < states.index(TableState.OCCUPIED)

    def test_passthrough_does_not_close_cycle(self):
        """
        НОВЫЙ ТЕСТ: человек прошёл мимо (APPROACH → EMPTY) не закрывает цикл.
        Это ключевое отличие от старой логики с мгновенным APPROACH.
        """
        m = make_monitor(empty=3, occupied=2, stay=5)

        # Сначала занимаем стол по-настоящему (нужно 5 кадров stay)
        feed(m, True, 2)           # → APPROACH
        feed(m, True, 5, start_frame=2)  # → OCCUPIED (stay=5)
        assert m.state == TableState.OCCUPIED

        feed(m, False, 3, start_frame=10)  # гость ушёл → EMPTY, цикл открыт
        assert m.state == TableState.EMPTY

        # Кто-то подошёл но сразу ушёл (APPROACH → EMPTY)
        feed(m, True, 2, start_frame=13)   # → APPROACH
        feed(m, False, 3, start_frame=15)  # → EMPTY (прошёл мимо, stay=5 не набрал)
        assert m.state == TableState.EMPTY

        a = m.get_analytics()
        assert a["completed_cycles"] == 0    # прохожий не засчитан
        assert a["open_cycles"] >= 1         # хотя бы один цикл остался открытым


# ---------------------------------------------------------------------------
# Группа 4: Аналитика
# ---------------------------------------------------------------------------

class TestAnalytics:

    def test_analytics_empty_when_no_cycles(self):
        """Без завершённых циклов аналитика возвращает None в числовых полях."""
        m = make_monitor()
        a = m.get_analytics()
        assert a["completed_cycles"] == 0
        assert a["mean_response_sec"] is None

    def test_response_time_calculated_correctly(self):
        """
        Время реакции = timestamp OCCUPIED − timestamp EMPTY.

        При FPS=10:
          EMPTY  на кадре 12 → 1.2 сек
          OCCUPIED на кадре 16 → 1.6 сек  (APPROACH на 14, +2 stay кадра)
          delta = 0.4 сек
        """
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)                 # кадры 0-3: первый гость
        feed(m, False, 3, start_frame=10)        # кадры 10-12: EMPTY на кадре 12
        feed(m, True, 2, start_frame=13)         # кадры 13-14: APPROACH на кадре 14
        feed(m, True, 2, start_frame=15)         # кадры 15-16: OCCUPIED на кадре 16

        a = m.get_analytics()
        assert a["completed_cycles"] == 1
        # EMPTY=1.2s, OCCUPIED=1.6s → delta=0.4s
        assert abs(a["mean_response_sec"] - 0.4) < 0.01

    def test_mean_over_multiple_cycles(self):
        """Среднее считается по всем завершённым циклам."""
        m = make_monitor(empty=3, occupied=2)
        frame = 0
        for _ in range(3):
            frame = occupy(m, n_occupied=2, start_frame=frame)
            feed(m, False, 3, frame); frame += 3
        # Последний подход — закрывает третий цикл
        feed(m, True, 2, frame); frame += 2
        feed(m, True, 2, frame)

        a = m.get_analytics()
        assert a["completed_cycles"] == 3
        assert a["mean_response_sec"] is not None

    def test_open_cycle_not_counted_in_mean(self):
        """Незакрытый цикл в среднее не входит."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)   # стол освободился — цикл открыт

        a = m.get_analytics()
        assert a["open_cycles"] == 1
        assert a["completed_cycles"] == 0
        assert a["mean_response_sec"] is None

    def test_analytics_keys_present(self):
        """get_analytics() всегда возвращает все ключи."""
        m = make_monitor()
        a = m.get_analytics()
        expected_keys = {
            "total_cycles", "completed_cycles", "open_cycles",
            "mean_response_sec", "median_response_sec",
            "min_response_sec", "max_response_sec",
        }
        assert set(a.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Группа 5: DataFrame-ы
# ---------------------------------------------------------------------------

class TestDataFrames:

    def test_events_dataframe_columns(self):
        m = make_monitor()
        df = m.get_events_dataframe()
        expected = {"frame_no", "timestamp_sec", "prev_state", "next_state", "event_name"}
        assert expected.issubset(set(df.columns))

    def test_events_dataframe_empty_when_no_transitions(self):
        m = make_monitor()
        df = m.get_events_dataframe()
        assert len(df) == 0

    def test_events_dataframe_has_correct_rows(self):
        """После полного цикла в DataFrame: APPROACH, OCCUPIED, EMPTY, APPROACH, OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        df = m.get_events_dataframe()
        states = list(df["next_state"])
        assert states[0] == "APPROACH"
        assert states[1] == "OCCUPIED"
        assert states[2] == "EMPTY"
        assert states[3] == "APPROACH"
        assert states[4] == "OCCUPIED"

    def test_events_dataframe_timestamp_monotonic(self):
        """Временны́е метки событий строго возрастают."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        df = m.get_events_dataframe()
        assert df["timestamp_sec"].is_monotonic_increasing

    def test_cycles_dataframe_columns(self):
        m = make_monitor()
        df = m.get_cycles_dataframe()
        expected = {
            "empty_at_frame", "empty_at_sec",
            "approach_at_frame", "approach_at_sec",
            "response_time_sec", "is_completed",
        }
        assert expected.issubset(set(df.columns))

    def test_cycles_dataframe_completed_flag(self):
        """is_completed=True для закрытых циклов, False для открытых."""
        m = make_monitor(empty=3, occupied=2)

        # Цикл 1 — закрытый (APPROACH подтверждён в OCCUPIED)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        # Цикл 2 — открытый (видео кончилось)
        feed(m, False, 3, start_frame=20)

        df = m.get_cycles_dataframe()
        assert len(df) == 2
        assert df[df["is_completed"]].shape[0] == 1
        assert df[~df["is_completed"]].shape[0] == 1

    def test_cycles_dataframe_response_time_matches_analytics(self):
        """response_time_sec в DataFrame совпадает с mean из get_analytics()."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        df = m.get_cycles_dataframe()
        a = m.get_analytics()
        df_mean = df["response_time_sec"].dropna().mean()
        assert abs(df_mean - a["mean_response_sec"]) < 0.01

    def test_events_dataframe_returns_copy(self):
        """transitions не открывает доступ к внутреннему состоянию."""
        m = make_monitor()
        t1 = m.transitions
        t1.append("injected")
        assert len(m.transitions) == 0


# ---------------------------------------------------------------------------
# Группа 6: Граничные случаи
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_never_occupied(self):
        """Всё видео стол пустой — никаких переходов."""
        m = make_monitor()
        feed(m, False, 100)
        assert m.state == TableState.EMPTY
        assert len(m.transitions) == 0

    def test_always_occupied_from_start(self):
        """Стол занят с первого кадра → APPROACH, потом OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)            # → APPROACH
        feed(m, True, 2, start_frame=2)  # → OCCUPIED
        assert m.state == TableState.OCCUPIED
        states = [t.next_state for t in m.transitions]
        assert states[0] == TableState.APPROACH
        assert states[1] == TableState.OCCUPIED

    def test_rapid_flicker_does_not_cause_false_transitions(self):
        """Чередование True/False каждый кадр не должно давать переходов."""
        m = make_monitor(empty=5, occupied=5)
        occupy(m, n_occupied=5)   # занять стол (через APPROACH, stay=5)
        frame = 10
        for i in range(20):
            m.update(frame + i, FPS, i % 2 == 0)
        assert m.state == TableState.OCCUPIED

    def test_single_frame_video(self):
        """Видео из одного кадра — не падает."""
        m = make_monitor(empty=3, occupied=2)
        result = m.update(0, FPS, False)
        assert result is None
        assert m.state == TableState.EMPTY

    def test_fps_affects_timestamp(self):
        """Временны́е метки зависят от fps, не захардкожены."""
        m1 = make_monitor(empty=3, occupied=2)
        m2 = make_monitor(empty=3, occupied=2)
        for i in range(2):
            m1.update(i, 10.0, True)
            m2.update(i, 30.0, True)
        ts1 = m1.transitions[0].timestamp
        ts2 = m2.transitions[0].timestamp
        assert abs(ts1 - 0.1) < 1e-9
        assert abs(ts2 - 1/30) < 1e-9

    def test_zero_response_time_impossible(self):
        """Время реакции не может быть нулевым."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)
        feed(m, True, 2, start_frame=15)

        df = m.get_cycles_dataframe()
        assert (df["response_time_sec"].dropna() > 0).all()

    def test_multiple_open_cycles_at_end(self):
        """Открытый цикл корректно считается при незавершённом видео."""
        m = make_monitor(empty=3, occupied=2)
        occupy(m, n_occupied=2)
        feed(m, False, 3, start_frame=10)   # open cycle 1 → EMPTY
        feed(m, True, 2, start_frame=13)    # → APPROACH
        feed(m, True, 2, start_frame=15)    # → OCCUPIED (closes cycle 1)
        feed(m, False, 3, start_frame=20)   # open cycle 2

        a = m.get_analytics()
        assert a["open_cycles"] == 1
        assert a["completed_cycles"] == 1


# ---------------------------------------------------------------------------
# Группа 7: Несколько циклов подряд
# ---------------------------------------------------------------------------

class TestMultipleCycles:

    @staticmethod
    def _run_n_cycles(n: int) -> TableMonitor:
        m = make_monitor(empty=3, occupied=2)
        frame = 0
        for _ in range(n):
            frame = occupy(m, n_occupied=2, start_frame=frame)
            feed(m, False, 3, frame); frame += 3
        # Последний подход — закрывает последний цикл
        feed(m, True, 2, frame); frame += 2
        feed(m, True, 2, frame)
        return m

    def test_three_cycles_completed(self):
        m = self._run_n_cycles(3)
        a = m.get_analytics()
        assert a["completed_cycles"] == 3

    def test_transitions_count_correct(self):
        """
        За 3 цикла должно быть:
          3 × (APPROACH + OCCUPIED) + 3 × EMPTY + 1 финальный APPROACH + OCCUPIED
          = 3*2 + 3 + 2 = 11 переходов.
        """
        m = self._run_n_cycles(3)
        df = m.get_events_dataframe()
        # Каждый цикл: APPROACH + OCCUPIED + EMPTY = 3 события
        # Финальный подход: APPROACH + OCCUPIED = 2 события
        # Итого: 3*3 + 2 = 11
        assert len(df) == 11

    def test_all_response_times_positive(self):
        m = self._run_n_cycles(5)
        df = m.get_cycles_dataframe()
        completed = df[df["is_completed"]]
        assert (completed["response_time_sec"] > 0).all()

    def test_cycles_dataframe_length(self):
        m = self._run_n_cycles(4)
        a = m.get_analytics()
        df = m.get_cycles_dataframe()
        assert len(df) == a["completed_cycles"] + a["open_cycles"]


# ---------------------------------------------------------------------------
# Группа 8: Специфические краевые случаи FSM
# ---------------------------------------------------------------------------

class TestFSMEdgeCases:

    def test_transition_chain_is_complete(self):
        """После APPROACH обязательно должно быть событие OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)            # → APPROACH
        feed(m, True, 2, start_frame=2)  # → OCCUPIED

        events = [t.next_state for t in m.transitions]
        assert TableState.APPROACH in events
        assert TableState.OCCUPIED in events
        assert events.index(TableState.APPROACH) < events.index(TableState.OCCUPIED)

    def test_no_dead_zone_after_approach(self):
        """После фиксации APPROACH система корректно начинает отсчёт до EMPTY."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)                 # → APPROACH (кадры 0-1)
        feed(m, True, 2, start_frame=2)  # → OCCUPIED (кадры 2-3)
        assert m.state == TableState.OCCUPIED

        transitions = feed(m, False, 3, start_frame=4)
        assert m.state == TableState.EMPTY
        assert any(t.next_state == TableState.EMPTY for t in transitions)

    def test_approach_is_atomic_event(self):
        """APPROACH не генерируется повторно, если стол не стал EMPTY."""
        m = make_monitor(empty=10, occupied=2)
        feed(m, True, 2)                 # → APPROACH
        feed(m, True, 2, start_frame=2)  # → OCCUPIED

        # Кратковременный дребезг (3 кадра, порог EMPTY = 10)
        feed(m, False, 3, start_frame=10)

        # Снова появился
        transitions = feed(m, True, 2, start_frame=13)

        approaches = [t for t in transitions if t.next_state == TableState.APPROACH]
        assert len(approaches) == 0


# ---------------------------------------------------------------------------
# Группа 9: Проверка инерции счётчиков (Debounce Integrity)
# ---------------------------------------------------------------------------

class TestDebounceIntegrity:

    def test_occupied_confirmation_requires_new_frames_after_approach(self):
        """
        После APPROACH счётчики сбрасываются.
        Следующий переход в OCCUPIED требует накопить min_stay_frames заново.
        """
        threshold = 5
        m = make_monitor(empty=10, occupied=threshold, stay=threshold)

        # Входим в APPROACH
        feed(m, True, threshold)
        assert m.state == TableState.APPROACH

        # Один кадр — недостаточно для OCCUPIED
        m.update(threshold, FPS, True)
        all_next_states = [t.next_state for t in m.transitions]
        assert TableState.OCCUPIED not in all_next_states

        # Добираем оставшиеся кадры
        feed(m, True, threshold - 1, start_frame=threshold + 1)
        all_next_states = [t.next_state for t in m.transitions]
        assert TableState.OCCUPIED in all_next_states

    def test_counter_reset_after_any_transition(self):
        """
        После перехода в APPROACH предыдущие накопленные кадры не должны
        засчитываться для перехода в OCCUPIED.
        """
        threshold = 5
        m = make_monitor(empty=10, occupied=threshold, stay=threshold)

        # Доходим до APPROACH
        feed(m, True, threshold)
        assert m.state == TableState.APPROACH

        # Если счётчик НЕ сбросился, одного кадра хватило бы для OCCUPIED
        m.update(threshold, FPS, True)

        # Проверяем: перехода в OCCUPIED ещё нет
        assert m.state == TableState.APPROACH

        # Добираем нужное количество кадров
        feed(m, True, threshold - 1, start_frame=threshold + 1)

        # Теперь переход должен произойти
        assert m.state == TableState.OCCUPIED

    def test_empty_counter_reset_after_transition(self):
        """
        После перехода в EMPTY предыдущие пустые кадры не должны влиять
        на следующий цикл (нужно заново накопить threshold).
        """
        m = make_monitor(empty=3, occupied=2)

        # Занимаем стол
        occupy(m, n_occupied=2)
        assert m.state == TableState.OCCUPIED

        # Переходим в EMPTY
        feed(m, False, 3, start_frame=10)
        assert m.state == TableState.EMPTY

        # Даём меньше порога для APPROACH
        transitions = feed(m, True, 1, start_frame=20)

        # Если счётчик не сброшен — уже был бы APPROACH
        assert m.state == TableState.EMPTY
        assert len(transitions) == 0

        # Добираем до порога
        transitions = feed(m, True, 1, start_frame=21)

        assert any(t.next_state == TableState.APPROACH for t in transitions)


# ---------------------------------------------------------------------------
# Группа 10: APPROACH как настоящее состояние (ключевые тесты новой логики)
# ---------------------------------------------------------------------------

class TestApproachAsRealState:

    def test_approach_to_occupied_transition_exists(self):
        """В истории должен быть переход APPROACH → OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)                 # → APPROACH
        feed(m, True, 2, start_frame=2)  # → OCCUPIED

        events = [(t.prev_state, t.next_state) for t in m.transitions]
        assert (TableState.APPROACH, TableState.OCCUPIED) in events

    def test_state_matches_last_transition(self):
        """Текущее состояние всегда равно последнему next_state в истории."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)                 # → APPROACH
        assert m.state == m.transitions[-1].next_state

        feed(m, True, 2, start_frame=2)  # → OCCUPIED
        assert m.state == m.transitions[-1].next_state

    def test_approach_persists_as_current_state(self):
        """APPROACH должен быть текущим состоянием в течение min_stay_frames кадров."""
        m = make_monitor(empty=3, occupied=2, stay=5)
        states_seen = set()
        for i in range(2):           # входим в APPROACH
            m.update(i, FPS, True)
            states_seen.add(m.state)
        for i in range(2, 6):        # наблюдаем APPROACH
            m.update(i, FPS, True)
            states_seen.add(m.state)

        assert TableState.APPROACH in states_seen

    def test_passthrough_returns_to_empty_not_occupied(self):
        """
        Человек прошёл мимо (ушёл до min_stay_frames) → EMPTY, не OCCUPIED.
        Это ключевое бизнес-правило нового APPROACH.
        """
        m = make_monitor(empty=3, occupied=2, stay=10)
        feed(m, True, 2)            # → APPROACH
        feed(m, False, 3, start_frame=2)  # прошёл мимо → EMPTY

        assert m.state == TableState.EMPTY
        states = [t.next_state for t in m.transitions]
        assert TableState.OCCUPIED not in states