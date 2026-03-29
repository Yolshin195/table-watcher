# conftest.py — общие фикстуры для всех тестов
import pytest
from src.table_monitor import TableMonitor

FPS = 10.0  # 1 кадр = 0.1 сек — удобные числа для проверки временных меток


@pytest.fixture
def make_monitor():
    """
    Фабрика мониторов с настраиваемыми порогами дебаунса.

    min_stay_frames по умолчанию равен min_occupied_frames —
    это значит, что ровно столько же кадров, сколько нужно для входа в APPROACH,
    нужно и для выхода из него в OCCUPIED. Старые тесты, написанные до введения
    APPROACH как отдельного состояния, продолжают работать без изменений.
    """
    def _factory(empty: int = 3, occupied: int = 2, stay: int = None) -> TableMonitor:
        return TableMonitor(
            min_empty_frames=empty,
            min_occupied_frames=occupied,
            min_stay_frames=stay if stay is not None else occupied,
        )
    return _factory


@pytest.fixture
def feed():
    """Вспомогательная функция: подать N одинаковых кадров подряд."""
    def _feed(monitor: TableMonitor, occupied: bool, n_frames: int, start_frame: int = 0):
        transitions = []
        for i in range(n_frames):
            t = monitor.update(start_frame + i, FPS, occupied)
            if t is not None:
                transitions.append(t)
        return transitions
    return _feed