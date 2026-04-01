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
import pandas as pd

from src.table_monitor import TableMonitor, TableState
from src.video_processor import BasePlugin, FrameContext
from src.utils.formatters import _fmt_dur, _fmt_ts
from src.cycles import TableCycle, build_cycles, cycles_to_dataframe

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
                rt = f"{row['response_time_sec']:.1f}s" if pd.notna(row["response_time_sec"]) else "—"
                status = "OK" if row["is_completed"] else "open"
                # approach_at_sec может быть None или NaN — _ts() теперь обрабатывает оба
                lines.append(
                    f"  empty={self._ts(row['empty_at_sec'])}"
                    f"  approach={self._ts(row['approach_at_sec'])}"
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
        # ИСПРАВЛЕНИЕ: pandas заменяет Python-None на float NaN при смешанных
        # типах в DataFrame. Проверка `sec is None` NaN не ловит — нужен pd.isna().
        if sec is None or (isinstance(sec, float) and pd.isna(sec)):
            return "—"
        m, s = divmod(float(sec), 60)
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
        # Добавляем текущее состояние в историю
        self._history.append(ctx.state)

        h, w = ctx.frame.shape[:2]
        if self._total_frames <= 0:
            return

        step = w / self._total_frames

        # ИСПРАВЛЕНИЕ O(n²) → O(1): рисуем только текущий сегмент,
        # а не всю историю с нуля на каждом кадре.
        i = len(self._history) - 1
        x_start = int(i * step)
        x_end   = max(x_start + 1, int((i + 1) * step))  # минимум 1px

        color = self._colors.get(ctx.state, (100, 100, 100))
        cv2.rectangle(
            ctx.frame,
            (x_start, h - self.bar_height),
            (x_end,   h),
            color,
            -1,
        )

        # Разделитель над таймлайном
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
    Рисует непрерывную ступенчатую линию с цветовой заливкой от t=0 до конца видео.
    """

    def __init__(self, output_path: Optional[Path | str], filename: str = "timeline.png"):
        self.output_dir = Path(output_path or ".")
        self.filename = filename
        self._total_frames: int = 0
        self._fps: float = 30.0

        self._colors_hex = {
            TableState.EMPTY:    '#00C800',
            TableState.OCCUPIED: '#DC0000',
            TableState.APPROACH: '#FFA500',
        }

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._total_frames = total_frames
        self._fps = fps

    def on_frame(self, ctx: FrameContext) -> None:
        pass

    def on_finish(self, monitor: TableMonitor) -> None:
        df = monitor.get_events_dataframe()
        analytics = monitor.get_analytics()

        if df.empty:
            logger.warning("TimelineChartPlugin: события не зафиксированы, график не будет создан.")
            return

        total_duration = self._total_frames / self._fps if self._fps > 0 else df['timestamp_sec'].max()

        # --- Определяем начальное состояние ---
        # Первое событие в df — это первый ПЕРЕХОД. Значит, до него было prev_state.
        initial_state = df.iloc[0]['prev_state']

        start_row = pd.DataFrame([{
            'timestamp_sec': 0.0,
            'next_state': initial_state,
        }])

        middle_rows = df[['timestamp_sec', 'next_state']].copy()

        end_row = pd.DataFrame([{
            'timestamp_sec': total_duration,
            'next_state': df.iloc[-1]['next_state'],  # последнее состояние держится до конца
        }])

        plot_df = pd.concat([start_row, middle_rows, end_row], ignore_index=True)

        # --- Построение графика ---
        fig, ax = plt.subplots(figsize=(15, 5))

        # 1. Ступенчатая линия
        ax.plot(
            plot_df['timestamp_sec'],
            plot_df['next_state'],
            drawstyle='steps-post',
            color='#222222',
            linewidth=2,
            zorder=4,
            label='Линия состояний',
        )

        # 2. Цветовая заливка интервалов (теперь покрывает весь диапазон [0, total_duration])
        for i in range(len(plot_df) - 1):
            t_start = plot_df.iloc[i]['timestamp_sec']
            t_end   = plot_df.iloc[i + 1]['timestamp_sec']
            state_name = plot_df.iloc[i]['next_state']

            try:
                state_enum = TableState[state_name]
                color = self._colors_hex.get(state_enum, '#999999')
            except KeyError:
                color = '#999999'

            ax.axvspan(t_start, t_end, color=color, alpha=0.25, zorder=2)

        # 3. Вертикальные маркеры событий (не считая искусственных точек)
        for _, row in df.iterrows():
            ax.axvline(x=row['timestamp_sec'], color='#555555', linewidth=0.8,
                       linestyle=':', alpha=0.7, zorder=3)

        # --- Оформление ---
        mean_val = analytics.get('mean_response_sec')
        title_text = "Хронология состояний столика"
        if mean_val:
            title_text += f"\nСреднее время между гостями: {mean_val:.1f}с"

        ax.set_title(title_text, fontsize=14, pad=15)
        ax.set_xlabel("Время видео (секунды)", fontsize=12)
        ax.set_ylabel("Состояние", fontsize=12)

        # Фиксируем диапазон X строго [0, total_duration]
        ax.set_xlim(0, total_duration)

        # Y-тики — все три состояния, даже если не все встречались
        all_states = [s.name for s in [TableState.APPROACH, TableState.EMPTY, TableState.OCCUPIED]]
        ax.set_yticks(all_states)

        ax.grid(axis='both', linestyle='--', alpha=0.4)

        # Легенда цветов состояний
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self._colors_hex[TableState.EMPTY],    alpha=0.5, label='EMPTY'),
            Patch(facecolor=self._colors_hex[TableState.APPROACH],  alpha=0.5, label='APPROACH'),
            Patch(facecolor=self._colors_hex[TableState.OCCUPIED],  alpha=0.5, label='OCCUPIED'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        fig.tight_layout()

        save_path = self.output_dir / self.filename
        fig.savefig(str(save_path), dpi=200)
        plt.close(fig)

        logger.info("TimelineChartPlugin: график сохранён в %s", save_path)


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


class BusinessAnalyticsPlugin(BasePlugin):
    """
    Финальный плагин отчета, полностью соответствующий ТЗ.
    Считает:
    1. Среднее время простоя (EMPTY -> APPROACH) — требование ТЗ.
    2. Среднее время обслуживания (длительность OCCUPIED).
    3. Статистику всех подходов (включая тех, кто не сел).
    """
    def __init__(
        self,
        output_path: Optional[Path | str],
        filename: str = "business_report.txt",
        video_path: str = ""
    ):
        self.output_path = Path(output_path or ".") / filename
        self.video_path = video_path
        self._roi = None

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._roi = roi
        self._fps = fps
    
    def on_frame(self, ctx: FrameContext) -> None:
        """
        Реализация абстрактного метода. 
        Нам не нужно обрабатывать каждый кадр, поэтому просто пропускаем.
        """
        pass

    def on_finish(self, monitor: TableMonitor) -> None:
        # Получаем данные из монитора
        df_events = monitor.get_events_dataframe()
        df_cycles = monitor.get_cycles_dataframe() # Здесь наши исправленные циклы

        if df_events.empty:
            logger.warning("Нет событий для формирования отчета.")
            return

        # --- 1. Расчет метрики ТЗ: EMPTY -> APPROACH ---
        # Берем все случаи, где зафиксирован подход (даже если цикл не завершен)
        wait_times = df_cycles['response_time_sec'].dropna().tolist()
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

        # --- 2. Расчет времени "Стол занят" (OCCUPIED) ---
        # Считаем разницу между входом в OCCUPIED и выходом из него
        occupied_durations = []
        occ_start = None
        for _, row in df_events.iterrows():
            if row['next_state'] == 'OCCUPIED':
                occ_start = row['timestamp_sec']
            elif row['prev_state'] == 'OCCUPIED' and occ_start is not None:
                occupied_durations.append(row['timestamp_sec'] - occ_start)
                occ_start = None
        
        avg_occupied = sum(occupied_durations) / len(occupied_durations) if occupied_durations else 0

        # --- 3. Статистика прохожих (EMPTY -> APPROACH -> EMPTY) ---
        # Это циклы, которые остались в статусе is_completed = False, но имеют approach_at
        false_approaches = df_cycles[(df_cycles['is_completed'] == False) & (df_cycles['approach_at_sec'].notna())]

        # Формируем текст отчета
        lines = [
            "======================================================",
            "        ИТОГОВЫЙ БИЗНЕС-ОТЧЕТ (ПО ТРЕБОВАНИЯМ ТЗ)",
            "======================================================",
            f"Видео файл:      {self.video_path}",
            f"Зона стола (ROI): x={self._roi[0]}, y={self._roi[1]}, w={self._roi[2]}, h={self._roi[3]}",
            "------------------------------------------------------",
            "1. ОСНОВНАЯ МЕТРИКА ТЗ",
            f"Среднее время между уходом гостя и подходом: {avg_wait:.2f} сек",
            f"Всего зафиксировано подходов к столу:        {len(wait_times)}",
            "",
            "2. АНАЛИТИКА ЗАЯТОСТИ",
            f"Среднее время, пока стол был ЗАНЯТ:          {avg_occupied:.2f} сек",
            f"Количество подтвержденных посадок:           {len(occupied_durations)}",
            "",
            "3. СТАТИСТИКА ПОДХОДОВ (Прохожие/Официанты)",
            f"Случаи 'Approach -> Empty' (не сели):        {len(false_approaches)}",
            "------------------------------------------------------",
            "ПОДРОБНЫЙ ЛОГ ЦИКЛОВ:",
            "Освободился | Подошли     | Задержка | Статус"
        ]

        for _, row in df_cycles.iterrows():
            empty_ts = self._fmt_ts(row['empty_at_sec'])
            appr_ts  = self._fmt_ts(row['approach_at_sec'])
            delay    = f"{row['response_time_sec']:>6.1f}s" if row['response_time_sec'] else "      --"
            status   = "ПОСАДКА" if row['is_completed'] else "ПРОХОЖИЙ"
            lines.append(f"{empty_ts}    | {appr_ts}    | {delay} | {status}")

        lines.append("======================================================")

        # Сохранение
        self.output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ Текстовый отчет по ТЗ сохранен в: {self.output_path}")

    @staticmethod
    def _fmt_ts(sec):
        if sec is None or pd.isna(sec): return "--:--:--"
        m, s = divmod(sec, 60)
        return f"{int(m):02d}:{s:05.2f}"


class IntervalAnalyticsPlugin(BasePlugin):
    """
    Плагин нового поколения. 
    Анализирует интервалы (Interval-based) вместо событийных хуков.
    """
    def __init__(
        self,
        output_path: Optional[Path | str],
        filename: str = "interval_report.txt",
    ):
        self.output_path = Path(output_path or ".") / filename
    
    def on_frame(self, ctx: FrameContext) -> None:
        """
        Реализация абстрактного метода. 
        Нам не нужно обрабатывать каждый кадр, поэтому просто пропускаем.
        """
        pass

    def on_finish(self, monitor: TableMonitor) -> None:
        # 1. Получаем все интервалы (включая текущий активный)
        df = monitor.get_intervals_dataframe()
        if df.empty:
            logger.warning("Нет данных для анализа интервалов.")
            return

        # 2. Логика поиска "Времени реакции":
        # Нам нужно найти пары: [EMPTY] -> [APPROACH или OCCUPIED]
        service_delays = []
        
        # Итерируемся по списку интервалов
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_int = df.iloc[i+1]
            
            # Если стол освободился (стал EMPTY)
            if current['state'] == "EMPTY":
                # Время "реакции" — это когда кто-то подошел (APPROACH)
                # или сразу сел (OCCUPIED), если APPROACH был пропущен
                if next_int['state'] in ["APPROACH", "OCCUPIED"]:
                    delay = next_int['start_sec'] - current['start_sec']
                    service_delays.append({
                        "freed_at": current['start_sec'],
                        "staff_arrived_at": next_int['start_sec'],
                        "delay": delay
                    })

        # 3. Формируем отчет
        self._save_report(df, service_delays)

    def _save_report(self, df_intervals: pd.DataFrame, delays: list):
        lines = [
            "======================================================",
            "   ОТЧЕТ НА ОСНОВЕ ИНТЕРВАЛЬНОЙ ЛОГИКИ (80/20)",
            "======================================================",
            f"Всего зафиксировано интервалов: {len(df_intervals)}",
            ""
        ]

        if delays:
            avg_delay = sum(d['delay'] for d in delays) / len(delays)
            lines.extend([
                "1. МЕТРИКА ОБСЛУЖИВАНИЯ",
                f"Среднее время до подхода к пустому столу: {avg_delay:.2f} сек",
                f"Количество циклов уборки/посадки: {len(delays)}",
                ""
            ])
        else:
            lines.append("Метрики задержки: Данных недостаточно (стол не освобождался или никто не подошел)\n")

        lines.append("2. ХРОНОЛОГИЯ ИНТЕРВАЛОВ:")
        lines.append("Старт (сек) | Конец (сек) | Состояние  | Длительность")
        lines.append("------------------------------------------------------")
        
        for _, row in df_intervals.iterrows():
            lines.append(
                f"{row['start_sec']:>11.2f} | {row['end_sec']:>11.2f} | "
                f"{row['state']:<10} | {row['duration']:>6.2f}s"
            )

        self.output_path.write_text("\n".join(lines), encoding="utf-8")
        
        logger.info(f"Интервальный отчет сохранен в: {self.output_path}")


class CsvIntervalExportPlugin(BasePlugin):
    """
    Плагин для экспорта всех накопленных интервалов состояний в CSV файл.
    Позволяет сохранить структурированные данные для последующего анализа в Excel/Pandas.
    """    
    def __init__(
        self,
        output_path: Optional[Path | str],
        filename: str = "intervals_data.csv",
    ):
        self.output_path = Path(output_path or ".") / filename
    
    def on_frame(self, ctx: FrameContext) -> None:
        """
        Реализация абстрактного метода. 
        Нам не нужно обрабатывать каждый кадр, поэтому просто пропускаем.
        """
        pass

    def on_finish(self, monitor: TableMonitor) -> None:
        """
        Вызывается один раз в конце обработки видео.
        Забирает данные из внутреннего контекста монитора через публичный метод.
        """
        # Используем встроенный в TableMonitor метод получения DataFrame
        df = monitor.get_intervals_dataframe()

        if df.empty:
            logger.warning("Экспорт в CSV: Интервалы не найдены, файл не будет создан.")
            return

        try:
            # Сохраняем в CSV
            # index=False, чтобы не плодить лишнюю колонку с ID строки
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"✅ Данные интервалов успешно экспортированы в CSV: {self.output_path}")
            print(f"  [CSV Export] Сохранено строк: {len(df)}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении CSV: {e}")

    def __repr__(self):
        return f"CsvIntervalExportPlugin(path='{self.output_path}')"


# ---------------------------------------------------------------------------
# TaskReportPlugin
# ---------------------------------------------------------------------------

class TaskReportPlugin(BasePlugin):
    """
    Единственный аналитический плагин для ТЗ.

    on_frame()  — ничего не делает (ноль overhead на горячем пути)
    on_finish() — строит циклы OCCUPIED→EMPTY→APPROACH постфактум,
                  считает метрики, сохраняет report.txt и cycles.csv

    Args:
        output_path: папка сессии
        video_path:  путь к видео (для отчёта)
        report_name: имя текстового отчёта (default: report.txt)
        csv_name:    имя CSV с циклами (default: cycles.csv)
    """

    def __init__(
        self,
        output_path: Optional[Path | str] = None,
        video_path:  str = "",
        report_name: str = "report.txt",
        csv_name:    str = "cycles.csv",
    ):
        self._base   = Path(output_path or ".")
        self._video  = video_path
        self._report = self._base / report_name
        self._csv    = self._base / csv_name

        self._roi:   Optional[tuple] = None
        self._total: int   = 0
        self._fps:   float = 30.0

    def on_start(self, total_frames: int, fps: float, roi: tuple) -> None:
        self._roi   = roi
        self._total = total_frames
        self._fps   = fps
        self._base.mkdir(parents=True, exist_ok=True)

    def on_frame(self, ctx: FrameContext) -> None:
        pass  # Вся логика в on_finish — никакого overhead

    def on_finish(self, monitor: TableMonitor) -> None:
        df_intervals = monitor.get_intervals_dataframe()

        if df_intervals.empty:
            logger.warning("TaskReportPlugin: интервалов нет — отчёт не создан.")
            return

        # Строим циклы OCCUPIED → EMPTY → APPROACH
        cycles    = build_cycles(df_intervals)
        df_cycles = cycles_to_dataframe(cycles)

        # Сохраняем
        self._save_csv(df_cycles)
        self._save_report(df_intervals, df_cycles, cycles)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _save_csv(self, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._csv, index=False, encoding="utf-8-sig")
            logger.info("TaskReportPlugin: CSV → %s  (%d циклов)", self._csv, len(df))
        except Exception as exc:
            logger.error("TaskReportPlugin: ошибка CSV: %s", exc)

    # ------------------------------------------------------------------
    # Текстовый отчёт
    # ------------------------------------------------------------------

    def _save_report(
        self,
        df_intervals: pd.DataFrame,
        df_cycles:    pd.DataFrame,
        cycles:       list[TableCycle],
    ) -> None:
        lines = self._build_report(df_intervals, df_cycles, cycles)
        try:
            self._report.write_text("\n".join(lines), encoding="utf-8")
            logger.info("TaskReportPlugin: отчёт → %s", self._report)
            print(f"\n✅  Отчёт ТЗ    → {self._report}")
            print(f"📊  Циклы CSV   → {self._csv}")
        except Exception as exc:
            logger.error("TaskReportPlugin: ошибка отчёта: %s", exc)

    def _build_report(
        self,
        df_intervals: pd.DataFrame,
        df_cycles:    pd.DataFrame,
        cycles:       list[TableCycle],
    ) -> list[str]:
        roi = self._roi or (0, 0, 0, 0)
        SEP  = "=" * 62
        THIN = "─" * 62

        complete   = [c for c in cycles if c.is_complete]
        incomplete = [c for c in cycles if not c.is_complete]
        wait_times = [c.wait_time for c in complete]

        # ── Шапка ────────────────────────────────────────────────────
        lines = [
            SEP,
            "   ОТЧЁТ: ДЕТЕКЦИЯ УБОРКИ СТОЛИКОВ  (требования ТЗ)",
            SEP,
            f"  Видео:    {self._video or '—'}",
            f"  ROI:      x={roi[0]}  y={roi[1]}  w={roi[2]}  h={roi[3]}",
            f"  Кадров:   {self._total}  |  FPS: {self._fps:.1f}",
            "",
        ]

        # ── 1. Основная метрика ТЗ ────────────────────────────────────
        lines += [
            "  1. ОСНОВНАЯ МЕТРИКА ТЗ",
            "  " + THIN,
            "  Цикл: OCCUPIED → EMPTY → APPROACH",
            "  Метрика: время пустоты между гостями (APPROACH.start − OCCUPIED.end)",
            "",
        ]

        if wait_times:
            avg    = sum(wait_times) / len(wait_times)
            median = sorted(wait_times)[len(wait_times) // 2]
            lines += [
                f"      Завершённых циклов : {len(complete)}",
                f"      Открытых циклов   : {len(incomplete)}  (нет следующего APPROACH)",
                "",
                f"      Среднее время пустоты  : {avg:.2f} сек",
                f"      Медиана                : {median:.2f} сек",
                f"      Минимум                : {min(wait_times):.2f} сек",
                f"      Максимум               : {max(wait_times):.2f} сек",
            ]
        else:
            lines += [
                f"      Завершённых циклов : 0",
                f"      Открытых циклов   : {len(incomplete)}",
                "      Данных недостаточно — нет полной тройки OCCUPIED→EMPTY→APPROACH",
            ]

        lines.append("")

        # ── 2. Детализация циклов ────────────────────────────────────
        lines += [
            "  2. ДЕТАЛИЗАЦИЯ ЦИКЛОВ  (OCCUPIED → EMPTY → APPROACH)",
            "  " + THIN,
            f"  {'№':>3}  {'Гость сел':>10}  {'Ушёл':>10}  "
            f"{'Сидел':>8}  {'Ждали':>8}  {'Подошли':>10}  Статус",
            "  " + "─" * 60,
        ]

        if not cycles:
            lines.append("  (циклов не зафиксировано)")
        else:
            for idx, c in enumerate(cycles, 1):
                status = "✓ полный" if c.is_complete else "○ открыт"
                lines.append(
                    f"  {idx:>3}.  "
                    f"{_fmt_ts(c.occupied_start_sec):>10}  "
                    f"{_fmt_ts(c.occupied_end_sec):>10}  "
                    f"{_fmt_dur(c.occupied_duration):>8}  "
                    f"{_fmt_dur(c.wait_time):>8}  "
                    f"{_fmt_ts(c.approach_start_sec):>10}  "
                    f"{status}"
                )

        lines.append("")

        # ── 3. Полная хронология интервалов ──────────────────────────
        lines += [
            "  3. ХРОНОЛОГИЯ ИНТЕРВАЛОВ",
            "  " + THIN,
            f"  {'Старт':>10}  {'Конец':>10}  {'Длит.':>8}  Состояние",
            "  " + "─" * 46,
        ]

        for _, row in df_intervals.iterrows():
            lines.append(
                f"  {_fmt_ts(row['start_sec']):>10}  "
                f"{_fmt_ts(row['end_sec']):>10}  "
                f"{row['duration']:>6.2f} s  "
                f"{row['state']}"
            )

        lines += ["", SEP]
        return lines

    def __repr__(self) -> str:
        return f"TaskReportPlugin(report='{self._report}', csv='{self._csv}')"