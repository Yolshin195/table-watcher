import pandas as pd
from src.table_monitor import TableState


# ---------------------------------------------------------------------------
# Группа: Корректная работа FSM (по ТЗ)
# ---------------------------------------------------------------------------

class TestFSMCorrectBehavior:

    def test_empty_to_approach_on_person_appearance(self, make_monitor, feed):
        """
        Если стол был EMPTY и появляется человек (после debounce),
        должен быть переход EMPTY → APPROACH.
        """
        m = make_monitor(empty=3, occupied=2)

        feed(m, False, 5)
        feed(m, True, 2)

        events = [(t.prev_state, t.next_state) for t in m.transitions]

        assert (TableState.EMPTY, TableState.APPROACH) in events


    def test_approach_then_becomes_occupied(self, make_monitor, feed):
        """
        APPROACH — это событие появления, после него состояние должно стать OCCUPIED.
        """
        m = make_monitor(empty=3, occupied=2)

        feed(m, False, 5)
        feed(m, True, 2)

        assert m.state == TableState.OCCUPIED


    def test_occupied_to_empty_after_person_leaves(self, make_monitor, feed):
        """
        Если человек ушёл и прошло достаточно кадров,
        должен быть переход OCCUPIED → EMPTY.
        """
        m = make_monitor(empty=3, occupied=2)

        feed(m, True, 3)
        feed(m, False, 3)

        events = [(t.prev_state, t.next_state) for t in m.transitions]

        assert (TableState.OCCUPIED, TableState.EMPTY) in events


    def test_no_false_transitions_due_to_noise(self, make_monitor, feed):
        """
        Дебаунс должен защищать от ложных срабатываний.
        Короткие всплески не должны менять состояние.
        """
        m = make_monitor(empty=5, occupied=5)

        feed(m, True, 1)

        assert len(m.transitions) == 0
        assert m.state == TableState.EMPTY


    def test_multiple_cycles_tracked_correctly(self, make_monitor, feed):
        """
        Несколько циклов EMPTY → APPROACH должны корректно фиксироваться.
        """
        m = make_monitor(empty=2, occupied=2)

        feed(m, True, 2)
        feed(m, False, 2)

        feed(m, True, 2)

        df = m.get_cycles_dataframe()

        assert len(df) >= 1
        assert df["is_completed"].sum() >= 1


    def test_response_time_positive(self, make_monitor, feed):
        """
        Время между EMPTY и APPROACH должно быть > 0.
        """
        m = make_monitor(empty=2, occupied=2)

        feed(m, True, 3)
        feed(m, False, 3)
        feed(m, True, 3)

        analytics = m.get_analytics()

        assert analytics["mean_response_sec"] is not None
        assert analytics["mean_response_sec"] > 0


    def test_events_dataframe_structure(self, make_monitor, feed):
        """
        DataFrame событий должен иметь правильную структуру.
        """
        m = make_monitor(empty=2, occupied=2)

        feed(m, True, 3)

        df = m.get_events_dataframe()

        assert set(df.columns) == {
            "frame_no", "timestamp_sec", "prev_state", "next_state", "event_name"
        }


    def test_cycles_dataframe_structure(self, make_monitor, feed):
        """
        DataFrame циклов должен иметь правильные поля.
        """
        m = make_monitor(empty=2, occupied=2)

        feed(m, True, 3)
        feed(m, False, 3)
        feed(m, True, 3)

        df = m.get_cycles_dataframe()

        assert "empty_at_sec" in df.columns
        assert "approach_at_sec" in df.columns
        assert "response_time_sec" in df.columns
        assert "is_completed" in df.columns