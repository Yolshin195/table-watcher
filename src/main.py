"""
Детекция уборки столиков по видео.

Запуск (минимальный):
    python main.py --video video1.mp4

Запуск с явным ROI (без интерактивного выбора):
    python main.py --video video1.mp4 --roi 120 80 240 160

Запуск без визуализации (фоновый режим, быстрее):
    python main.py --video video1.mp4 --no-overlay

Полный пример:
    python main.py --video video1.mp4 \
        --output output.mp4           \
        --report report.txt           \
        --snapshots snapshots/        \
        --confidence 0.45             \
        --empty-frames 25             \
        --occupied-frames 4           \
        --model yolov8n.pt
"""

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Настройка логирования — до любых других импортов
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Парсинг аргументов
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Детекция уборки столиков по видео с камеры пиццерии.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Обязательный аргумент ---
    parser.add_argument(
        "--video", "-v",
        required=True,
        metavar="PATH",
        help="Путь к входному видеофайлу (mp4, avi, …)",
    )

    # --- Зона столика ---
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help=(
            "Координаты зоны столика: x y w h (в пикселях). "
            "Если не задано — откроется интерактивный выбор через cv2.selectROI."
        ),
    )

    # --- Выходные файлы ---
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default="output.mp4",
        help="Путь для сохранения видео с визуализацией (default: output.mp4).",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default="report.txt",
        help="Путь для текстового отчёта (default: report.txt).",
    )
    parser.add_argument(
        "--snapshots",
        metavar="DIR",
        default="snapshots",
        help=(
            "Директория для PNG-скриншотов в момент событий FSM "
            "(default: snapshots/). Передай '' чтобы отключить."
        ),
    )

    # --- Детектор ---
    parser.add_argument(
        "--model",
        metavar="NAME",
        default="yolov8n.pt",
        help="Имя YOLO-модели (default: yolov8n.pt). Варианты: yolov8n/s/m/l/x.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        metavar="FLOAT",
        help="Минимальная уверенность детектора [0.0–1.0] (default: 0.4).",
    )

    # --- Параметры FSM ---
    parser.add_argument(
        "--empty-frames",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Сколько кадров подряд зона должна быть пустой, "
            "чтобы зафиксировать событие EMPTY (default: 30 ≈ 1 сек при 30fps)."
        ),
    )
    parser.add_argument(
        "--occupied-frames",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Сколько кадров подряд в зоне должен быть человек, "
            "чтобы зафиксировать событие OCCUPIED (default: 5)."
        ),
    )

    # --- Режим работы ---
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help=(
            "Фоновый режим: не рисовать overlay на кадрах. "
            "Видео не записывается. Только аналитика и отчёт."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Не показывать прогресс-бар в консоли.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод (DEBUG-уровень).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Показывать видео в окне во время обработки. "
            "Управление: Q/ESC — выход, ПРОБЕЛ — пауза."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Масштаб окна просмотра [0.1–2.0] (default: 1.0).",
    )

    return parser


# ---------------------------------------------------------------------------
# Валидация аргументов
# ---------------------------------------------------------------------------

def _validate(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        print(f"Ошибка: файл не найден: {video}", file=sys.stderr)
        sys.exit(1)
    if not video.is_file():
        print(f"Ошибка: путь не является файлом: {video}", file=sys.stderr)
        sys.exit(1)

    if args.roi is not None:
        x, y, w, h = args.roi
        if w <= 0 or h <= 0:
            print(f"Ошибка: ширина и высота ROI должны быть > 0, получено w={w} h={h}", file=sys.stderr)
            sys.exit(1)

    if not (0.0 < args.confidence <= 1.0):
        print(f"Ошибка: --confidence должен быть в диапазоне (0, 1], получено {args.confidence}", file=sys.stderr)
        sys.exit(1)

    if args.empty_frames < 1:
        print(f"Ошибка: --empty-frames должен быть >= 1", file=sys.stderr)
        sys.exit(1)

    if args.occupied_frames < 1:
        print(f"Ошибка: --occupied-frames должен быть >= 1", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Сборка плагинов согласно аргументам
# ---------------------------------------------------------------------------

def _build_plugins(args: argparse.Namespace) -> list:
    from plugins import (
        RoiOverlayPlugin,
        EventLoggerPlugin,
        ProgressPlugin,
        SnapshotPlugin,
        ReportPlugin,
        TimelinePlugin,
    )

    plugins = []

    # Overlay рисуется только если не фоновый режим
    if not args.no_overlay:
        plugins.append(RoiOverlayPlugin())
        plugins.append(TimelinePlugin())

    # Live-просмотр — подключается ПОСЛЕ overlay, чтобы видеть отрисовку
    if args.live:
        from plugins import LiveViewPlugin
        plugins.append(LiveViewPlugin(scale=args.scale))

    # Лог событий FSM в консоль — всегда
    plugins.append(EventLoggerPlugin())

    # Прогресс-бар
    if not args.no_progress:
        plugins.append(ProgressPlugin())

    # Скриншоты в момент событий
    if args.snapshots:
        plugins.append(SnapshotPlugin(output_dir=args.snapshots))

    # Текстовый отчёт
    plugins.append(ReportPlugin(
        output_path=args.report,
        video_path=args.video,
    ))

    return plugins


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)
    _validate(args)

    log = logging.getLogger(__name__)

    # Импортируем тяжёлые зависимости после валидации
    from table_monitor import TableMonitor
    from video_processor import VideoProcessor

    roi = tuple(args.roi) if args.roi else None

    monitor = TableMonitor(
        min_empty_frames=args.empty_frames,
        min_occupied_frames=args.occupied_frames,
    )

    plugins = _build_plugins(args)

    # В фоновом режиме output не пишем
    output_path = None if args.no_overlay else args.output

    log.info("Видео:   %s", args.video)
    log.info("Модель:  %s  (confidence=%.2f)", args.model, args.confidence)
    log.info("FSM:     empty_frames=%d  occupied_frames=%d",
             args.empty_frames, args.occupied_frames)
    if roi:
        log.info("ROI:     x=%d y=%d w=%d h=%d", *roi)
    else:
        log.info("ROI:     интерактивный выбор")

    processor = VideoProcessor(
        video_path=args.video,
        roi=roi,
        monitor=monitor,
        plugins=plugins,
        confidence_threshold=args.confidence,
        output_path=output_path,
        model_name=args.model,
    )

    monitor = processor.run()

    # Итоговая сводка в консоль
    analytics = monitor.get_analytics()
    print()
    print("─" * 50)
    print(f"  Циклов завершено : {analytics['completed_cycles']}")
    print(f"  Циклов открытых  : {analytics['open_cycles']}")

    mean = analytics["mean_response_sec"]
    if mean is not None:
        print(f"  Среднее время    : {mean:.1f} сек")
        print(f"  Медиана          : {analytics['median_response_sec']:.1f} сек")
        print(f"  Мин / Макс       : {analytics['min_response_sec']:.1f} / {analytics['max_response_sec']:.1f} сек")
    else:
        print("  Среднее время    : n/a (нет завершённых циклов)")

    if output_path:
        print(f"  Видео сохранено  : {output_path}")
    print(f"  Отчёт сохранён   : {args.report}")
    if args.snapshots:
        print(f"  Скриншоты        : {args.snapshots}/")
    print("─" * 50)


if __name__ == "__main__":
    main()