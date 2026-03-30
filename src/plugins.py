"""
Встроенные плагины для VideoProcessor.

Подключай любой набор в зависимости от задачи:

    from plugins import RoiOverlayPlugin, EventLoggerPlugin, ProgressPlugin, ReportPlugin

    processor = VideoProcessor("video.mp4", plugins=[
        RoiOverlayPlugin(),
        EventLoggerPlugin(),
        ProgressPlugin(),
        ReportPlugin("report.txt"),
    ])
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from table_monitor import TableMonitor, TableState
from video_processor import BasePlugin, FrameContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Цвета состояний (BGR)
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[TableState, tuple[int,int,int]] = {
    TableState.EMPTY:    (0, 200, 0),    # зелёный
    TableState.OCCUPIED: (0, 0, 220),    # красный
    TableState.APPROACH: (0, 165, 255),  # оранжевый
}

_STATE_LABELS: dict[TableState, str] = {
    TableState.EMPTY:    "EMPTY",
    TableState.OCCUPIED: "OCCUPIED",
    TableState.APPROACH: "APPROACH",
}


# ---------------------------------------------------------------------------
# 1. RoiOverlayPlugin — рисует bbox зоны столика и оверлей состояния
# ---------------------------------------------------------------------------

class RoiOverlayPlugin(BasePlugin):
    """
    Рисует на кадре:
    - Прямоугольник зоны ROI (цвет зависит от состояния FSM)
    - Метку состояния над прямоугольником
    - Таймер сколько секунд стол в текущем состоянии
    - Опционально: bounding box детекции людей (если show_detections=True)
    """

    def __init__(self, show_detections: bool = False, thickness: int = 2):
        self.show_detections = show_detections
        self.thickness       = thickness
        self._state_since_sec: float = 0.0

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._state_since_sec = 0.0
        self._last_state: Optional[TableState] = None

    def on_frame(self, ctx: FrameContext) -> None:
        # Обновляем таймер при смене состояния
        if ctx.transition is not None:
            self._state_since_sec = ctx.timestamp_sec
        elapsed = ctx.timestamp_sec - self._state_since_sec

        color  = _STATE_COLORS[ctx.state]
        label  = _STATE_LABELS[ctx.state]
        x1, y1, x2, y2 = ctx.roi_rect

        # Прямоугольник ROI
        cv2.rectangle(ctx.frame, (x1, y1), (x2, y2), color, self.thickness)

        # Метка состояния
        text = f"{label}  {elapsed:.1f}s"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(ctx.frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            ctx.frame, text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def on_finish(self, monitor: TableMonitor) -> None:
        pass


# ---------------------------------------------------------------------------
# 2. PeopleVisualizerPlugin — Рисует рамки и точки детекции, используя PersonDetection
# ---------------------------------------------------------------------------
class PeopleVisualizerPlugin(BasePlugin):
    """Рисует рамки и точки детекции, используя PersonDetection."""

    def on_frame(self, ctx: FrameContext) -> None:
        for person in ctx.detected_people:
            x1, y1, x2, y2 = person.bbox
            
            # Красный для тех, кто в ROI, Зеленый для остальных
            color = (0, 0, 255) if person.is_in_roi else (0, 255, 0)
            thickness = 2 if person.is_in_roi else 1

            # Отрисовка BBox
            cv2.rectangle(ctx.frame, (x1, y1), (x2, y2), color, thickness)
            
            # Отрисовка "точки опоры" (ног)
            cv2.circle(ctx.frame, person.foot_point, 4, color, -1)
            
            # Подпись уверенности
            label = f"{person.confidence:.2f}"
            cv2.putText(
                ctx.frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )


# ---------------------------------------------------------------------------
# 3. EventLoggerPlugin — логирует каждое событие FSM в консоль
# ---------------------------------------------------------------------------
class EventLoggerPlugin(BasePlugin):
    """
    Выводит в лог каждый StateTransition в момент его возникновения.

    Пример вывода:
        [00:01:23.4 | frame 2081]  OCCUPIED → EMPTY
        [00:01:47.1 | frame 2513]  EMPTY    → APPROACH  (+23.7s)
    """

    def __init__(self, level: int = logging.INFO):
        self._level = level
        self._last_empty_sec: Optional[float] = None

    def _safe_log(self, message: str, *args):
        """Печатает лог, очищая текущую строку (важно для работы с \r)"""
        # Очищаем строку в терминале перед выводом лога
        sys.stdout.write("\r" + " " * 120 + "\r")
        logger.log(self._level, message, *args)

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._last_empty_sec = None

    def on_frame(self, ctx: FrameContext) -> None:
        if ctx.transition is None:
            return

        t = ctx.transition
        ts = self._fmt_time(t.timestamp)

        if t.next_state == TableState.EMPTY:
            self._last_empty_sec = t.timestamp
            self._safe_log("[%s | frame %d]  %s", ts, t.frame_no, t.event_name)

        elif t.next_state == TableState.APPROACH and self._last_empty_sec is not None:
            delta = t.timestamp - self._last_empty_sec
            self._safe_log(
                "[%s | frame %d]  %s  (+%.1fs after empty)",
                ts, t.frame_no, t.event_name, delta,
            )
            self._last_empty_sec = None
        else:
            self._safe_log("[%s | frame %d]  %s", ts, t.frame_no, t.event_name)

    def on_finish(self, monitor: TableMonitor) -> None:
        a = monitor.get_analytics()
        self._safe_log("─" * 50)
        self._safe_log("Завершено. Циклов: %d  |  Среднее время реакции: %s",
                   a["completed_cycles"],
                   f"{a['mean_response_sec']:.1f}s" if a["mean_response_sec"] else "n/a")

    @staticmethod
    def _fmt_time(sec: float) -> str:
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# 4. ProgressPlugin — прогресс-бар в консоли
# ---------------------------------------------------------------------------

class ProgressPlugin(BasePlugin):
    """
    Печатает прогресс обработки в одну строку (перезаписывает её через \r).
    Обновляется раз в update_every кадров чтобы не тормозить.
    """

    def __init__(self, update_every: int = 30, bar_width: int = 40):
        self.update_every = update_every
        self.bar_width    = bar_width
        self._total: int  = 0
        self._fps: float  = 30.0
        self._t0: float   = 0.0

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._total = total_frames
        self._fps   = fps
        self._t0    = time.monotonic()

    def on_frame(self, ctx: FrameContext) -> None:
        if ctx.frame_no % self.update_every != 0:
            return

        elapsed  = time.monotonic() - self._t0
        progress = ctx.frame_no / max(self._total, 1)
        filled   = int(self.bar_width * progress)
        bar      = "█" * filled + "░" * (self.bar_width - filled)
        
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        ts  = ctx.timestamp_sec

        # Формируем строку. \r возвращает курсор в начало.
        # Конец строки дополняем пробелами, чтобы затереть старые хвосты если они были.
        output = (
            f"\r[{bar}] {progress*100:5.1f}% | "
            f"video:{ts:>6.1f}s | wall:{int(elapsed):>4}s | "
            f"ETA:{int(eta):>4}s | {ctx.state.name:<9}"
        )
        sys.stdout.write(output)
        sys.stdout.flush()

    def on_finish(self, monitor: TableMonitor) -> None:
        elapsed = time.monotonic() - self._t0
        # Очищаем строку полностью перед финальным сообщением
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.write(f"✅ Обработка завершена за {elapsed:.1f}s\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# 5. SnapshotPlugin — сохраняет кадр при каждой смене состояния
# ---------------------------------------------------------------------------

class SnapshotPlugin(BasePlugin):
    """
    Сохраняет PNG-скриншот кадра в момент каждого StateTransition.
    Удобно для README: «пример проблемного кадра».

    Файлы: {output_dir}/frame_{frame_no:06d}_{event}.png
    """

    def __init__(self, output_path: Optional[Path | str] = None):
        self.output_dir = Path(output_path or ".") / "snapshots"

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_frame(self, ctx: FrameContext) -> None:
        if ctx.transition is None:
            return

        event_tag = f"{ctx.transition.prev_state.name}_to_{ctx.transition.next_state.name}"
        filename  = self.output_dir / f"frame_{ctx.frame_no:06d}_{event_tag}.png"
        cv2.imwrite(str(filename), ctx.frame)
        logger.debug("Snapshot: %s", filename)

    def on_finish(self, monitor: TableMonitor) -> None:
        snaps = list(self.output_dir.glob("*.png"))
        logger.info("SnapshotPlugin: сохранено %d скриншотов в %s", len(snaps), self.output_dir)


# ---------------------------------------------------------------------------
#6. ReportPlugin — записывает текстовый отчёт после обработки
# ---------------------------------------------------------------------------

class ReportPlugin(BasePlugin):
    """
    После обработки всего видео записывает текстовый отчёт в файл.

    Формат:
        === Отчёт по видео ===
        Видео:    video1.mp4
        ROI:      x=120 y=80 w=240 h=160
        ...
        Среднее время реакции: 34.2 сек
        Медиана:               28.5 сек
        ...
        === События ===
        00:01:23.40  OCCUPIED → EMPTY
        00:01:47.10  EMPTY    → APPROACH  (+23.7s)
    """

    def __init__(self, output_path: Optional[Path | str] = None, video_path: str = ""):
        self.output_path = Path(output_path or ".") / "report.txt"
        self.video_path  = video_path
        self._roi: Optional[tuple] = None

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._roi        = roi
        self._total      = total_frames
        self._fps        = fps

    def on_frame(self, ctx: FrameContext) -> None:
        pass  # всё делаем в on_finish

    def on_finish(self, monitor: TableMonitor) -> None:
        a   = monitor.get_analytics()
        df  = monitor.get_events_dataframe()
        cyc = monitor.get_cycles_dataframe()

        lines = [
            "=" * 54,
            "  Отчёт: детекция уборки столиков",
            "=" * 54,
            f"  Видео:          {self.video_path or '—'}",
            f"  ROI:            x={self._roi[0]} y={self._roi[1]} "
            f"w={self._roi[2]} h={self._roi[3]}",
            f"  Всего кадров:   {self._total}",
            f"  FPS:            {self._fps:.1f}",
            "",
            "  Аналитика",
            "  " + "─" * 36,
            f"  Циклов завершено:  {a['completed_cycles']}",
            f"  Циклов открытых:   {a['open_cycles']}",
            f"  Среднее время:     {self._fmt(a['mean_response_sec'])}",
            f"  Медиана:           {self._fmt(a['median_response_sec'])}",
            f"  Минимум:           {self._fmt(a['min_response_sec'])}",
            f"  Максимум:          {self._fmt(a['max_response_sec'])}",
            "",
            "  События",
            "  " + "─" * 36,
        ]

        for _, row in df.iterrows():
            lines.append(
                f"  {self._ts(row['timestamp_sec'])}"
                f"  {row['prev_state']:<8} → {row['next_state']}"
            )

        if not cyc.empty:
            lines += ["", "  Циклы (стол освободился → подход)", "  " + "─" * 36]
            for _, row in cyc.iterrows():
                rt = f"{row['response_time_sec']:.1f}s" if row["response_time_sec"] else "—"
                status = "OK" if row["is_completed"] else "open"
                lines.append(
                    f"  empty={self._ts(row['empty_at_sec'])}"
                    f"  approach={self._ts(row['approach_at_sec']) if row['approach_at_sec'] else '—'}"
                    f"  delta={rt}  [{status}]"
                )

        lines.append("=" * 54)

        self.output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Отчёт сохранён: %s", self.output_path)

    @staticmethod
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.2f} сек" if val is not None else "n/a"

    @staticmethod
    def _ts(sec: Optional[float]) -> str:
        if sec is None:
            return "—"
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# 7. LiveViewPlugin — показывает кадры в окне cv2.imshow в реальном времени
# ---------------------------------------------------------------------------
 
class LiveViewPlugin(BasePlugin):
    """
    Открывает окно cv2.imshow и показывает каждый обработанный кадр.
 
    Важно: вызывать ПОСЛЕ RoiOverlayPlugin в списке плагинов —
    тогда в окне будет виден уже нарисованный overlay.
 
    Управление:
        Q / ESC  — прервать обработку досрочно
        ПРОБЕЛ   — пауза / продолжить
 
    Args:
        window_title : заголовок окна
        scale        : масштаб отображения (1.0 = оригинальный размер,
                       0.5 = половина — удобно для больших видео)
        wait_ms      : задержка между кадрами в мс (1 = максимальная скорость,
                       увеличь для замедленного просмотра)
    """
 
    class _StopRequested(Exception):
        """Сигнал досрочного выхода — перехватывается VideoProcessor."""
        pass
 
    def __init__(
        self,
        window_title: str = "Table Monitor",
        scale:        float = 1.0,
        wait_ms:      int = 1,
    ):
        self.window_title = window_title
        self.scale        = scale
        self.wait_ms      = wait_ms
        self._paused      = False
        self._stopped     = False
 
    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        logger.info(
            "LiveViewPlugin: окно '%s' открыто  [Q/ESC=выход  ПРОБЕЛ=пауза]",
            self.window_title,
        )
 
    def on_frame(self, ctx: FrameContext) -> None:
        if self._stopped:
            return
 
        frame = ctx.frame
        if self.scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_LINEAR,
            )
 
        cv2.imshow(self.window_title, frame)
 
        # Обработка клавиш — пауза блокирует до следующего нажатия
        while True:
            key = cv2.waitKey(self.wait_ms) & 0xFF
 
            if key in (ord("q"), ord("Q"), 27):   # Q или ESC
                logger.info("LiveViewPlugin: пользователь прервал обработку")
                self._stopped = True
                raise LiveViewPlugin._StopRequested()
 
            if key == ord(" "):                    # пробел — пауза/продолжить
                self._paused = not self._paused
                logger.info("LiveViewPlugin: %s", "пауза" if self._paused else "продолжить")
 
            if not self._paused:
                break
 
    def on_finish(self, monitor: TableMonitor) -> None:
        cv2.destroyWindow(self.window_title)


# ---------------------------------------------------------------------------
# 8. TimelinePlugin — отрисовывает историю состояний внизу кадра
# ---------------------------------------------------------------------------

class TimelinePlugin(BasePlugin):
    """
    Отрисовывает динамический таймлайн в нижней части кадра.
    Цвет полоски соответствует состоянию стола в конкретный момент времени.
    """

    def __init__(self, bar_height: int = 35):
        self.bar_height = bar_height
        self._total_frames: int = 0
        # Храним историю состояний для отрисовки всей полоски на каждом кадре
        self._history: list[TableState] = []
        
        # Используем те же цвета, что и в основной системе
        self._colors = {
            TableState.EMPTY:    (0, 255, 0),    # Зеленый
            TableState.OCCUPIED: (0, 0, 255),    # Красный
            TableState.APPROACH: (0, 255, 255),  # Желтый/Оранжевый
        }

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._total_frames = total_frames
        self._history = []

    def on_frame(self, ctx: FrameContext) -> None:
        # 1. Запоминаем текущее состояние для истории
        self._history.append(ctx.state)
        
        # 2. Логика отрисовки таймлайна
        h, w = ctx.frame.shape[:2]
        
        if self._total_frames <= 0:
            return

        # Рассчитываем ширину шага для одного кадра
        step = w / self._total_frames
        
        # Отрисовываем накопленную историю
        # Для оптимизации: если видео очень длинное, можно рисовать не каждый кадр,
        # а группировать их, но при текущей логике рисуем всё накопленное.
        for i, state in enumerate(self._history):
            x_start = int(i * step)
            x_end = int((i + 1) * step)
            
            color = self._colors.get(state, (100, 100, 100))
            
            # Рисуем сегмент таймлайна внизу кадра
            cv2.rectangle(
                ctx.frame, 
                (x_start, h - self.bar_height), 
                (x_end, h), 
                color, 
                -1
            )
            
        # Опционально: рисуем тонкую рамку или разделитель над таймлайном
        cv2.line(ctx.frame, (0, h - self.bar_height), (w, h - self.bar_height), (255, 255, 255), 1)

    def on_finish(self, monitor: TableMonitor) -> None:
        # Очищаем историю после завершения, чтобы не держать память
        self._history.clear()


# ---------------------------------------------------------------------------
# 9. TimelineChartPlugin — генерирует PNG-график истории состояний
# ---------------------------------------------------------------------------

class TimelineChartPlugin(BasePlugin):
    """
    Плагин для генерации финального аналитического графика.
    
    Использует данные из TableMonitor после завершения обработки видео.
    Рисует хронологию переходов и выводит среднее время реакции.
    """

    def __init__(self, output_path: Optional[Path], filename: str = "timeline.png"):
        self.output_dir = Path(output_path or ".")
        self.filename = filename
        
        # Цвета для matplotlib (совпадают по логике с основной системой)
        self._colors_hex = {
            TableState.EMPTY:    '#00C800',  # Зеленый
            TableState.OCCUPIED: '#DC0000',  # Красный
            TableState.APPROACH: '#FFA500',  # Оранжевый
        }

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        # Создаем папку для отчетов заранее
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_frame(self, ctx: FrameContext) -> None:
        # В процессе работы видео ничего не делаем, чтобы не тратить ресурсы
        pass

    def on_finish(self, monitor: TableMonitor) -> None:
        """
        Основная логика: берем накопленные данные из монитора и строим график.
        """
        df = monitor.get_events_dataframe()
        analytics = monitor.get_analytics()

        if df.empty:
            logger.warning("TimelineChartPlugin: события не зафиксированы, график не будет создан.")
            return

        # Подготовка фигуры
        plt.figure(figsize=(15, 6))
        
        # Рисуем точки для каждого состояния из колонки 'next_state'
        # В вашем мониторе это зафиксированные переходы (StateTransition)
        unique_states = df['next_state'].unique()
        
        for state_name in unique_states:
            subset = df[df['next_state'] == state_name]
            
            # Получаем цвет через Enum (state_name там строка, например 'OCCUPIED')
            try:
                state_enum = TableState[state_name]
                color = self._colors_hex.get(state_enum, 'gray')
            except KeyError:
                color = 'gray'
            
            plt.scatter(
                subset['timestamp_sec'], 
                [state_name] * len(subset), 
                c=color, 
                label=state_name, 
                s=30, 
                marker='|',
                zorder=3
            )

        # Формируем заголовок с аналитикой
        mean_val = analytics.get('mean_response_sec')
        title_text = "Хронология состояний столика"
        if mean_val:
            title_text += f"\nСреднее время между гостями: {mean_val:.1f}с"

        plt.title(title_text, fontsize=14, pad=15)
        plt.xlabel("Время видео (секунды)", fontsize=12)
        plt.ylabel("Состояние", fontsize=12)
        
        # Сетка и оформление
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.yticks(unique_states)
        plt.legend(frameon=True, loc='upper right')
        
        plt.tight_layout()
        
        # Сохранение
        save_path = self.output_dir / self.filename
        plt.savefig(str(save_path), dpi=200)
        plt.close()
        
        logger.info("TimelineChartPlugin: Аналитический график сохранен в %s", save_path)


class TableProgressBarPlugin(BasePlugin):
    """
    Плагин для визуализации прогресса накопления кадров (дебаунса) в реальном времени.
    
    Отображает три горизонтальных индикатора, которые показывают, насколько текущее 
    состояние счетчиков близко к срабатыванию бизнес-правил перехода:
    - **APPROACH**: Прогресс появления человека в зоне (Empty -> Approach).
    - **OCCUPIED**: Прогресс подтверждения гостя (Approach -> Occupied).
    - **EMPTY**: Прогресс очистки стола (Occupied/Approach -> Empty).

    Цветовая схема синхронизирована с глобальными константами состояний системы.
    """

    def __init__(self, x: int = 30, y: int = 30, width: int = 220):
        """
        Инициализирует плагин отрисовки прогресс-баров.

        Args:
            x (int): Координата X верхнего левого угла блока индикаторов.
            y (int): Координата Y верхнего левого угла блока индикаторов.
            width (int): Ширина полосок в пикселях.
        """
        self.x = x
        self.y = y
        self.width = width
        self.bar_h = 12   # Высота одной полоски
        self.spacing = 25 # Расстояние между полосками (включая место под текст)

    def on_frame(self, ctx: FrameContext) -> None:
        """
        Отрисовывает прогресс-бары на текущем кадре видео.

        Данные берутся из `ctx.progress`, который содержит нормализованные 
        значения (от 0.0 до 1.0), рассчитанные в `TableMonitor`.

        Args:
            ctx (FrameContext): Объект контекста кадра, содержащий `np.ndarray` 
                изображения и Snapshot прогресса.
        """
        # Связываем ключи прогресса с цветами и названиями из общих констант
        bars = [
            ("to_approach", TableState.APPROACH),
            ("to_occupied", TableState.OCCUPIED),
            ("to_empty",    TableState.EMPTY),
        ]

        for i, (progress_key, state_type) in enumerate(bars):
            # Получаем значение прогресса (защита от отсутствия ключа)
            value = getattr(ctx.progress, progress_key, 0.0)
            
            # Получаем общие цвета и метки
            color = _STATE_COLORS.get(state_type, (255, 255, 255))
            label = _STATE_LABELS.get(state_type, "UNKNOWN")
            
            # Вычисляем вертикальную позицию текущего элемента
            curr_y = self.y + i * (self.bar_h + self.spacing)
            
            # --- 1. Текст (Заголовок над полоской) ---
            # Черная обводка для читаемости на любом фоне
            cv2.putText(ctx.frame, label, (self.x, curr_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            # Белый текст с процентами
            cv2.putText(ctx.frame, f"{label}: {int(value*100)}%", (self.x, curr_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # --- 2. Подложка полоски (Фон) ---
            cv2.rectangle(ctx.frame, (self.x, curr_y), 
                          (self.x + self.width, curr_y + self.bar_h), (30, 30, 30), -1)
            
            # --- 3. Активный прогресс ---
            fill_w = int(self.width * value)
            if fill_w > 0:
                cv2.rectangle(ctx.frame, (self.x, curr_y), 
                              (self.x + fill_w, curr_y + self.bar_h), color, -1)
                
            # --- 4. Контур (Рамка) ---
            cv2.rectangle(ctx.frame, (self.x, curr_y), 
                          (self.x + self.width, curr_y + self.bar_h), (150, 150, 150), 1)


class UnifiedHistoryLogger(BasePlugin):
    """
    Финальная версия логгера для ТЗ.
    Фиксирует: Старт -> Все переходы -> Аналитику задержек.
    Показывает: % прогресса, Frame, Video Time, Wall Time, ETA и Processing FPS.
    """
    def __init__(self, bar_width: int = 20):
        self.bar_width = bar_width
        self._total_frames = 0
        self._t0 = 0.0
        self._last_empty_sec = None
        self._first_frame = True

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._total_frames = total_frames
        self._t0 = time.monotonic()
        print("\n" + "="*95)
        print(f"ROI: {roi} | TOTAL FRAMES: {total_frames}")
        print("="*95 + "\n")

    def _build_line(self, ctx: FrameContext, state_text: str) -> str:
        elapsed = time.monotonic() - self._t0
        progress = ctx.frame_no / max(self._total_frames, 1)
        
        # Расчет FPS обработки (сколько кадров в секунду обрабатывает ПК)
        proc_fps = ctx.frame_no / elapsed if elapsed > 0 else 0
        
        # Визуальный бар
        filled = int(self.bar_width * progress)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        
        # Сборка строки: [Бар] % | frame | video | wall | ETA | FPS | Status
        return (
            f"[{bar}] {progress*100:4.1f}% | "
            f"frame: {ctx.frame_no:>5} | "
            f"video:{ctx.timestamp_sec:>6.1f}s | "
            f"wall:{int(elapsed):>3}s | "
            f"ETA:{int(eta):>3}s | "
            f"{proc_fps:>4.1f} fps | "
            f"{state_text}"
        )

    def on_frame(self, ctx: FrameContext) -> None:
        # 1. Фиксируем ТОЧКУ СТАРТА (первый кадр видео)
        if self._first_frame:
            line = self._build_line(ctx, f"{ctx.state.name}")
            sys.stdout.write(f"\r{line}\n")
            self._first_frame = False
            return

        # 2. ФИКСАЦИЯ ПЕРЕХОДА (в историю)
        if ctx.transition:
            t = ctx.transition
            msg = f"{t.prev_state.name} → {t.next_state.name}"
            
            if t.next_state == TableState.APPROACH and self._last_empty_sec is not None:
                delta = ctx.timestamp_sec - self._last_empty_sec
                msg += f" (+{delta:.1f}s)"
            
            if t.next_state == TableState.EMPTY:
                self._last_empty_sec = ctx.timestamp_sec

            line = self._build_line(ctx, msg)
            sys.stdout.write(f"\r{line}\n")
            sys.stdout.flush()
            return

        # 3. ЖИВОЕ ОБНОВЛЕНИЕ (каждые 5 кадров)
        if ctx.frame_no % 5 == 0:
            line = self._build_line(ctx, f"Status: {ctx.state.name}")
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()

    def on_finish(self, monitor: TableMonitor) -> None:
        a = monitor.get_analytics()
        print("\n" + "─"*95)
        if a['mean_response_sec']:
            print(f"📊 СРЕДНЕЕ ВРЕМЯ МЕЖДУ ГОСТЯМИ (ТЗ): {a['mean_response_sec']} сек")
        print("="*95 + "\n")