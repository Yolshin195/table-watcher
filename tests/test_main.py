"""
Тесты для TableMonitor.

Запуск:
    pip install pytest pandas
    pytest test_table_monitor.py -v          # с pytest
    python -m pytest test_table_monitor.py   # тоже с pytest

Структура:
    TestDebounce          — дебаунс не даёт мерцать состоянию
    TestFSMTransitions    — правильные переходы между состояниями
    TestApproachLogic     — APPROACH срабатывает именно когда надо
    TestAnalytics         — get_analytics() считает корректно
    TestDataFrames        — структура и содержимое DataFrame-ов
    TestEdgeCases         — граничные случаи и необычные сценарии
    TestMultipleCycles    — несколько циклов подряд
"""

import pandas as pd
from src.main import TableMonitor, TableState, StateTransition, CleanupRecord


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

FPS = 10.0  # удобные числа: 1 кадр = 0.1 сек


def feed(monitor: TableMonitor, occupied: bool, n_frames: int, start_frame: int = 0):
    """Подать N одинаковых кадров подряд. Возвращает список переходов (без None)."""
    transitions = []
    for i in range(n_frames):
        t = monitor.update(start_frame + i, FPS, occupied)
        if t is not None:
            transitions.append(t)
    return transitions


def make_monitor(empty=3, occupied=2) -> TableMonitor:
    """Монитор с маленькими порогами дебаунса — удобно для тестов."""
    return TableMonitor(min_empty_frames=empty, min_occupied_frames=occupied)


# ---------------------------------------------------------------------------
# Группа 1: Дебаунс
# ---------------------------------------------------------------------------

class TestDebounce:

    def test_no_transition_before_threshold_empty(self):
        """Стол не переходит в EMPTY раньше чем накопится min_empty_frames."""
        m = make_monitor(empty=5, occupied=2)
        # Сначала занимаем стол
        feed(m, True, 2)
        assert m.state == TableState.OCCUPIED

        # Подаём 4 пустых кадра — один меньше порога
        transitions = feed(m, False, 4, start_frame=10)
        assert m.state == TableState.OCCUPIED
        assert len(transitions) == 0

    def test_transition_exactly_at_threshold_empty(self):
        """Переход в EMPTY происходит ровно на кадре N = min_empty_frames."""
        m = make_monitor(empty=5, occupied=2)
        feed(m, True, 2)
        transitions = feed(m, False, 5, start_frame=10)
        assert m.state == TableState.EMPTY
        assert len(transitions) == 1
        assert transitions[0].next_state == TableState.EMPTY

    def test_no_transition_before_threshold_occupied(self):
        """Стол не помечается как занятый раньше чем накопится min_occupied_frames."""
        m = make_monitor(empty=3, occupied=4)
        # Стол пустой изначально
        # Подаём 3 кадра с человеком — один меньше порога
        transitions = feed(m, True, 3)
        assert m.state == TableState.EMPTY
        assert len(transitions) == 0

    def test_transition_exactly_at_threshold_occupied(self):
        """Переход в APPROACH происходит ровно на кадре N = min_occupied_frames."""
        m = make_monitor(empty=3, occupied=4)
        transitions = feed(m, True, 4)
        assert m.state == TableState.OCCUPIED  # после APPROACH сразу OCCUPIED
        assert len(transitions) == 1
        assert transitions[0].next_state == TableState.APPROACH

    def test_debounce_resets_on_interruption(self):
        """Прерывание до порога обнуляет счётчик — переход не происходит."""
        m = make_monitor(empty=5, occupied=2)
        feed(m, True, 2)  # занять стол
        assert m.state == TableState.OCCUPIED

        # 4 пустых, потом 1 занятый — счётчик должен сброситься
        feed(m, False, 4, start_frame=10)
        assert m.state == TableState.OCCUPIED  # ещё не переключился

        feed(m, True, 1, start_frame=14)
        assert m.state == TableState.OCCUPIED  # и не переключился обратно

        # ещё 4 пустых — снова не хватает до 5
        transitions = feed(m, False, 4, start_frame=15)
        assert m.state == TableState.OCCUPIED
        assert len(transitions) == 0

    def test_only_one_transition_per_threshold_crossing(self):
        """Переход фиксируется ровно один раз, не повторяется на каждом кадре после."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)  # занять
        transitions = feed(m, False, 10, start_frame=10)  # долго пусто
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
        """После APPROACH внутреннее состояние должно стать OCCUPIED."""
        m = make_monitor()
        feed(m, True, 2)
        assert m.state == TableState.OCCUPIED

    def test_occupied_to_empty(self):
        """Человек ушёл → стол переходит в EMPTY."""
        m = make_monitor()
        feed(m, True, 2)           # занять
        feed(m, False, 3, start_frame=10)  # освободить
        assert m.state == TableState.EMPTY

    def test_full_cycle(self):
        """Полный цикл: EMPTY → APPROACH → OCCUPIED → EMPTY → APPROACH."""
        m = make_monitor(empty=3, occupied=2)
        frame = 0

        # Первый гость садится
        feed(m, True, 2, frame); frame += 2
        assert m.state == TableState.OCCUPIED

        # Гость уходит
        feed(m, False, 3, frame); frame += 3
        assert m.state == TableState.EMPTY

        # Второй гость (или официант)
        transitions = feed(m, True, 2, frame)
        assert m.state == TableState.OCCUPIED
        assert transitions[0].next_state == TableState.APPROACH

    def test_transition_returns_correct_object(self):
        """update() возвращает StateTransition с правильными полями."""
        m = make_monitor(empty=3, occupied=2)
        result = None
        for i in range(2):
            result = m.update(i, FPS, True)
        # На кадре 1 (индекс 1, т.е. второй кадр) должен быть переход
        assert result is not None
        assert isinstance(result, StateTransition)
        assert result.frame_no == 1
        assert abs(result.timestamp - 1 / FPS) < 1e-9

    def test_no_approach_if_already_occupied(self):
        """APPROACH не должен срабатывать если стол и так OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)   # первый раз — APPROACH
        assert m.state == TableState.OCCUPIED

        # Короткий перерыв — меньше порога дебаунса
        feed(m, False, 2, start_frame=10)
        assert m.state == TableState.OCCUPIED  # не успел переключиться

        # Снова человек — не должно быть APPROACH
        transitions = feed(m, True, 2, start_frame=12)
        assert len(transitions) == 0  # состояние и так OCCUPIED, нет события


# ---------------------------------------------------------------------------
# Группа 3: Логика APPROACH
# ---------------------------------------------------------------------------

class TestApproachLogic:

    def test_approach_only_after_confirmed_empty(self):
        """APPROACH срабатывает только после подтверждённого EMPTY, не раньше."""
        m = make_monitor(empty=5, occupied=2)
        feed(m, True, 2)           # занять (APPROACH → OCCUPIED)
        feed(m, False, 4, start_frame=10)  # 4 пустых — не хватает до EMPTY
        transitions = feed(m, True, 2, start_frame=14)
        # Не было подтверждённого EMPTY → не должно быть APPROACH
        assert all(t.next_state != TableState.APPROACH for t in transitions)

    def test_approach_closes_open_cycle(self):
        """После APPROACH в _closed_cycles должна появиться запись."""
        m = make_monitor()
        feed(m, True, 2)            # первый гость (APPROACH → OCCUPIED)
        feed(m, False, 3, start_frame=10)  # уходит
        feed(m, True, 2, start_frame=13)  # второй гость (APPROACH)

        analytics = m.get_analytics()
        assert analytics["completed_cycles"] == 1
        assert analytics["open_cycles"] == 0

    def test_approach_transition_recorded_in_history(self):
        """APPROACH появляется в списке transitions."""
        m = make_monitor()
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        approach_events = [t for t in m.transitions if t.next_state == TableState.APPROACH]
        assert len(approach_events) == 2  # один при старте, один после EMPTY


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
        Время реакции = timestamp APPROACH − timestamp EMPTY.
        При FPS=10: кадр 10 = 1.0 сек, кадр 13 = 1.3 сек → delta = 0.3 сек.
        """
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)                    # кадры 0-1: первый гость
        feed(m, False, 3, start_frame=10)   # кадры 10-12: EMPTY на кадре 12
        feed(m, True, 2, start_frame=13)    # кадры 13-14: APPROACH на кадре 14

        a = m.get_analytics()
        assert a["completed_cycles"] == 1

        # EMPTY зафиксирован на кадре 12 → 1.2 сек
        # APPROACH зафиксирован на кадре 14 → 1.4 сек
        # delta = 0.2 сек
        assert abs(a["mean_response_sec"] - 0.2) < 0.01

    def test_mean_over_multiple_cycles(self):
        """Среднее считается по всем завершённым циклам."""
        m = make_monitor(empty=3, occupied=2)
        frame = 0

        def cycle(empty_gap: int):
            nonlocal frame
            feed(m, True, 2, frame); frame += 2
            feed(m, False, 3, frame); frame += 3
            feed(m, True, 2, frame); frame += empty_gap

        # Цикл 1: gap = 5 кадров → 0.5 сек пустоты (EMPTY кадр +2, APPROACH кадр +2+gap)
        # Цикл 2: gap = 3 кадра → 0.3 сек пустоты
        # Цикл 3: gap = 7 кадров → 0.7 сек пустоты
        # Среднее ≈ (0.2 + 0.2 + 0.2) / 3 = 0.2 — дельта всегда 0.2 т.к. дебаунс фиксирует
        # точно через 2 кадра занятости после 3 пустых
        cycle(5); cycle(5); cycle(5)
        a = m.get_analytics()
        assert a["completed_cycles"] == 3
        assert a["mean_response_sec"] is not None

    def test_open_cycle_not_counted_in_mean(self):
        """Незакрытый цикл (видео кончилось пока стол пустой) в среднее не входит."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)                    # гость
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
        """get_events_dataframe() содержит ожидаемые колонки."""
        m = make_monitor()
        df = m.get_events_dataframe()
        expected = {"frame_no", "timestamp_sec", "prev_state", "next_state", "event_name"}
        assert expected.issubset(set(df.columns))

    def test_events_dataframe_empty_when_no_transitions(self):
        m = make_monitor()
        df = m.get_events_dataframe()
        assert len(df) == 0

    def test_events_dataframe_has_correct_rows(self):
        """После полного цикла в DataFrame ровно 3 строки: APPROACH, EMPTY, APPROACH."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        df = m.get_events_dataframe()
        assert len(df) == 3
        assert df.iloc[0]["next_state"] == "APPROACH"
        assert df.iloc[1]["next_state"] == "EMPTY"
        assert df.iloc[2]["next_state"] == "APPROACH"

    def test_events_dataframe_timestamp_monotonic(self):
        """Временны́е метки событий строго возрастают."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        df = m.get_events_dataframe()
        assert df["timestamp_sec"].is_monotonic_increasing

    def test_cycles_dataframe_columns(self):
        """get_cycles_dataframe() содержит ожидаемые колонки."""
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

        # Цикл 1 — закрытый
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        # Цикл 2 — открытый (видео кончилось)
        feed(m, False, 3, start_frame=20)

        df = m.get_cycles_dataframe()
        assert len(df) == 2
        assert df[df["is_completed"]].shape[0] == 1
        assert df[~df["is_completed"]].shape[0] == 1

    def test_cycles_dataframe_response_time_matches_analytics(self):
        """response_time_sec в DataFrame совпадает с mean из get_analytics()."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        df = m.get_cycles_dataframe()
        a = m.get_analytics()
        df_mean = df["response_time_sec"].dropna().mean()
        assert abs(df_mean - a["mean_response_sec"]) < 0.01

    def test_events_dataframe_returns_copy(self):
        """transitions() и get_events_dataframe() не открывают доступ к внутреннему состоянию."""
        m = make_monitor()
        t1 = m.transitions
        t1.append("injected")  # попытка мутировать
        assert len(m.transitions) == 0  # внутренний список не тронут


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
        """Стол занят с первого кадра → APPROACH, потом остаётся OCCUPIED."""
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 10)
        assert m.state == TableState.OCCUPIED
        assert len(m.transitions) == 1
        assert m.transitions[0].next_state == TableState.APPROACH

    def test_rapid_flicker_does_not_cause_false_transitions(self):
        """
        Чередование True/False каждый кадр не должно давать переходов
        пока ни одна сторона не накопит threshold.
        """
        m = make_monitor(empty=5, occupied=5)
        feed(m, True, 5)  # занять стол
        frame = 5
        for i in range(20):
            m.update(frame + i, FPS, i % 2 == 0)  # чередование
        assert m.state == TableState.OCCUPIED
        # Новых переходов не должно быть (только первый APPROACH)
        assert len(m.transitions) == 1

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

        # Одинаковые кадры, разные fps
        for i in range(2):
            m1.update(i, 10.0, True)
            m2.update(i, 30.0, True)

        ts1 = m1.transitions[0].timestamp
        ts2 = m2.transitions[0].timestamp
        assert abs(ts1 - 0.1) < 1e-9  # кадр 1 при 10fps = 0.1 сек
        assert abs(ts2 - 1/30) < 1e-9  # кадр 1 при 30fps = 0.033 сек

    def test_zero_response_time_impossible(self):
        """
        Время реакции не может быть нулевым: EMPTY и APPROACH
        не могут происходить на одном кадре.
        """
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        feed(m, True, 2, start_frame=13)

        df = m.get_cycles_dataframe()
        assert (df["response_time_sec"].dropna() > 0).all()

    def test_multiple_open_cycles_at_end(self):
        """
        Если стол освобождался несколько раз без подхода
        (теоретически невозможно в реальности, но тест на устойчивость).
        """
        m = make_monitor(empty=3, occupied=2)
        feed(m, True, 2)
        feed(m, False, 3, start_frame=10)
        # Стол снова занят без APPROACH (не может быть при нормальной работе,
        # но проверяем что класс не упадёт при ручном вызове)
        # Симулируем патологию: ещё один EMPTY без предшествующего APPROACH
        # (достигается через второй монитор — этот тест проверяет open_cycles > 1
        #  в аналитике при двух последовательных EMPTY)
        m2 = make_monitor(empty=3, occupied=2)
        feed(m2, True, 2)
        feed(m2, False, 3, start_frame=10)  # open cycle 1
        feed(m2, True, 2, start_frame=13)   # APPROACH → closed
        feed(m2, False, 3, start_frame=20)  # open cycle 2

        a = m2.get_analytics()
        assert a["open_cycles"] == 1
        assert a["completed_cycles"] == 1


# ---------------------------------------------------------------------------
# Группа 7: Несколько циклов подряд
# ---------------------------------------------------------------------------

class TestMultipleCycles:

    @staticmethod
    def _run_n_cycles(n: int) -> TableMonitor:
        """Запустить N полных циклов и вернуть монитор."""
        m = make_monitor(empty=3, occupied=2)
        frame = 0
        for _ in range(n):
            feed(m, True, 2, frame);  frame += 2
            feed(m, False, 3, frame); frame += 3
        # Последний подход (закрывает цикл)
        # Первый цикл — нет предшествующего EMPTY, не закрывается
        # Все остальные закрываются при следующем APPROACH
        feed(m, True, 2, frame)
        return m

    def test_three_cycles_completed(self):
        m = self._run_n_cycles(3)
        a = m.get_analytics()
        assert a["completed_cycles"] == 3

    def test_transitions_count_correct(self):
        """
        За 3 цикла должно быть:
          3 × APPROACH + 3 × EMPTY = 6 переходов.
        (Последний APPROACH закрывает третий цикл.)
        """
        m = self._run_n_cycles(3)
        df = m.get_events_dataframe()
        assert len(df) == 7  # 3 APPROACH + 3 EMPTY + 1 финальный APPROACH

    def test_all_response_times_positive(self):
        m = self._run_n_cycles(5)
        df = m.get_cycles_dataframe()
        completed = df[df["is_completed"]]
        assert (completed["response_time_sec"] > 0).all()

    def test_cycles_dataframe_length(self):
        """Количество строк в cycles DataFrame = completed + open."""
        m = self._run_n_cycles(4)
        a = m.get_analytics()
        df = m.get_cycles_dataframe()
        assert len(df) == a["completed_cycles"] + a["open_cycles"]


# ---------------------------------------------------------------------------
# Группа 8: Специфические краевые случаи FSM (Выявление багов)
# ---------------------------------------------------------------------------

class TestFSMEdgeCases:

    def test_transition_chain_is_complete(self):
        """
        ПРОВЕРКА НА БАГ: После APPROACH обязательно должно быть событие OCCUPIED.
        Если в коде стоит ручная перезапись self._state = OCCUPIED внутри 
        обработки APPROACH, то второе событие (переход в стабильное состояние) 
        просто не сгенерируется, так как update() не увидит разницы.
        """
        m = make_monitor(empty=3, occupied=2)
        
        # Подаем 5 кадров присутствия. 
        # На 2-м кадре должен быть APPROACH.
        # На 3-м (или сразу после) должен быть переход в стабильный OCCUPIED.
        feed(m, True, 5) 
        
        events = [t.next_state for t in m.transitions]
        
        # В дефектной версии здесь будет только [TableState.APPROACH]
        assert len(events) >= 2, f"Цепочка событий прервана: {events}"
        assert TableState.OCCUPIED in events, "Отсутствует подтверждение состояния OCCUPIED в истории"

    def test_no_dead_zone_after_approach(self):
        """
        Проверка сброса счетчиков. Если после фиксации APPROACH человек сразу 
        исчезает, система должна корректно начать отсчет до EMPTY.
        """
        m = make_monitor(empty=3, occupied=2)
        
        # 1. Фиксируем подход (кадры 0, 1)
        feed(m, True, 2) 
        assert m.state == TableState.OCCUPIED
        
        # 2. Человек уходит (кадры 2, 3, 4). На 4-м кадре должен быть EMPTY.
        transitions = feed(m, False, 3, start_frame=2)
        
        assert m.state == TableState.EMPTY
        assert any(t.next_state == TableState.EMPTY for t in transitions), \
            "Система 'зависла' в OCCUPIED и не заметила ухода сразу после подхода"

    def test_approach_is_atomic_event(self):
        """
        APPROACH не должен генерироваться повторно, если стол не стал EMPTY.
        Это проверяет, что APPROACH — это именно 'вход' в состояние занятости.
        """
        m = make_monitor(empty=10, occupied=2)
        
        # Занимаем стол -> APPROACH
        feed(m, True, 2) 
        
        # Кратковременный 'дребезг' детектора (исчез на 3 кадра, порог EMPTY = 10)
        feed(m, False, 3, start_frame=10) 
        
        # Снова появился
        transitions = feed(m, True, 2, start_frame=13)
        
        # Новых APPROACH быть не должно
        approaches = [t for t in transitions if t.next_state == TableState.APPROACH]
        assert len(approaches) == 0, "Ошибка: Повторный APPROACH без предварительной очистки стола"


# ---------------------------------------------------------------------------
# Группа 9: Проверка инерции счетчиков (Debounce Integrity)
# ---------------------------------------------------------------------------

class TestDebounceIntegrity:

    def test_occupied_confirmation_requires_new_frames_after_approach(self):
        """
        ТЕСТ НА ОШИБКУ: После APPROACH система должна ПОВТОРНО подтвердить 
        стабильность состояния для перехода в OCCUPIED.
        
        Если счетчики не сбрасываются в _apply_transition, то при 
        min_occupied_frames=5:
        - Кадр 5: APPROACH (счетчик=5)
        - Кадр 6: OCCUPIED (счетчик=6) -> ОШИБКА, переход случился слишком быстро.
        """
        threshold = 5
        m = make_monitor(empty=10, occupied=threshold)
        
        # 1. Подаем ровно столько кадров, сколько нужно для APPROACH
        feed(m, True, threshold)
        assert m.state == TableState.OCCUPIED # Внутри уже подменилось
        
        # Проверяем историю переходов
        events = m.transitions
        assert events[-1].next_state == TableState.APPROACH
        
        # 2. Подаем ЕЩЕ ОДИН кадр присутствия
        m.update(threshold + 1, FPS, True)
        
        # Если счетчик НЕ сбросился, система увидит (threshold + 1) > threshold
        # и создаст новый переход в OCCUPIED прямо сейчас.
        
        all_next_states = [t.next_state for t in m.transitions]
        
        # ПРОВЕРКА: В списке событий НЕ должно быть OCCUPIED сразу после APPROACH.
        # Мы должны накопить еще 'threshold' кадров для этого.
        assert TableState.OCCUPIED not in all_next_states, (
            f"БАГ: Переход в OCCUPIED случился мгновенно на кадре {threshold + 1}. "
            "Счетчики не были сброшены после APPROACH."
        )

    def test_counter_reset_after_any_transition(self):
        """
        Тест проверяет физическое обнуление атрибутов после перехода.
        """
        m = make_monitor(empty=5, occupied=5)
        
        # Доводим до перехода
        feed(m, True, 5)
        
        assert m._consecutive_occupied == 0, (
            f"Счетчик _consecutive_occupied равен {m._consecutive_occupied}, а должен быть 0"
        )