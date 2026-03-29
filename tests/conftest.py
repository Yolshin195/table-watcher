import pytest
from src.table_monitor import TableMonitor

FPS = 10.0


@pytest.fixture
def fps():
    return FPS


@pytest.fixture
def make_monitor():
    def _make(empty=3, occupied=2):
        return TableMonitor(
            min_empty_frames=empty,
            min_occupied_frames=occupied
        )
    return _make


@pytest.fixture
def feed(fps):
    def _feed(monitor, occupied: bool, n_frames: int, start_frame: int = 0):
        transitions = []
        for i in range(n_frames):
            t = monitor.update(start_frame + i, fps, occupied)
            if t is not None:
                transitions.append(t)
        return transitions
    return _feed