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

from src.utils.roi_manager import ROIManager
from src.utils.session_dir import get_session_dir


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
        "--output-path",
        metavar="DIR",
        default="outputs",
        help="Корневая папка для всех запусков (default: outputs)."
    )
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
        "--step",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Запускать детектор только на каждом N-м кадре. "
            "Остальные кадры используют последний результат. "
            "step=3 даёт ~3× ускорение без потери точности меток (default: 1)."
        ),
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
        default=200,
        metavar="N",
        help=(
            "Порог очистки: сколько кадров подряд в зоне не должно быть людей, "
            "чтобы стол перешел в состояние EMPTY (защита от мерцания детектора)."
        ),
    )
    parser.add_argument(
        "--occupied-frames",
        type=int,
        default=15,
        metavar="N",
        help=(
            "Порог фиксации подхода: сколько кадров присутствия нужно, чтобы "
            "перейти из EMPTY в APPROACH и начать наблюдение за потенциальным гостем."
        ),
    )
    parser.add_argument(
        "--stay-frames",
        type=int,
        default=150,
        metavar="N",
        help=(
            "Порог подтверждения посадки: сколько кадров человек должен "
            "непрерывно находиться в зоне APPROACH, чтобы система зафиксировала "
            "состояние OCCUPIED и закрыла аналитический цикл ожидания."
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
    
    if args.stay_frames < 1:
        print(f"Ошибка: --stay_frames должен быть >= 1", file=sys.stderr)
        sys.exit(1)        


# ---------------------------------------------------------------------------
# Сборка плагинов согласно аргументам
# ---------------------------------------------------------------------------

def _build_plugins(args: argparse.Namespace, output_path: Path) -> list:
    from src.plugins import (
        RoiOverlayPlugin,
        EventLoggerPlugin,
        SnapshotPlugin,
        ReportPlugin,
        TimelinePlugin,
        TimelineChartPlugin,
        PeopleVisualizerPlugin,
        TableProgressBarPlugin,
        UnifiedHistoryLogger,
        IntervalAnalyticsPlugin,
        CsvIntervalExportPlugin,
        TaskReportPlugin,
    )

    plugins = []

    # Overlay рисуется только если не фоновый режим
    if not args.no_overlay:
        plugins.append(RoiOverlayPlugin())
        plugins.append(PeopleVisualizerPlugin())
        plugins.append(TimelinePlugin())
        plugins.append(TableProgressBarPlugin())

    # Live-просмотр — подключается ПОСЛЕ overlay, чтобы видеть отрисовку
    if args.live:
        from src.plugins import LiveViewPlugin
        plugins.append(LiveViewPlugin(scale=args.scale))

    # Скриншоты в момент событий
    if args.snapshots:
        plugins.append(SnapshotPlugin(output_path=output_path))

    plugins.append(IntervalAnalyticsPlugin(output_path=output_path))
    plugins.append(CsvIntervalExportPlugin(output_path=output_path))
    plugins.append(TaskReportPlugin(output_path=output_path))

    # Текстовый отчёт
    plugins.append(ReportPlugin(
        output_path=output_path,
        video_path=args.video,
    ))

    # Генерации финального аналитического графика.
    plugins.append(TimelineChartPlugin(output_path=output_path))

    # Прогресс-бар
    if not args.no_progress:
        plugins.append(UnifiedHistoryLogger())
    else:
        #Если прогресс бар не запущен лог событий FSM в консоль
        plugins.append(EventLoggerPlugin())
    
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
    from src.table_monitor import TableMonitor
    from src.video_processor import VideoProcessor

    roi = tuple(args.roi) if args.roi else None

    monitor = TableMonitor(
        min_empty_frames=args.empty_frames,
        min_occupied_frames=args.occupied_frames,
        min_stay_frames=args.stay_frames,
    )

    output_path = get_session_dir(args.output_path, args.video)
    plugins = _build_plugins(args, output_path)

    # В фоновом режиме output не пишем
    output_video = None if args.no_overlay else output_path / args.output

    log.info("Видео:   %s", args.video)
    log.info("Модель:  %s  (confidence=%.2f  step=%d)", args.model, args.confidence, args.step)
    log.info("FSM:     empty_frames=%d  occupied_frames=%d, stay_frames=%d",
             args.empty_frames, args.occupied_frames, args.stay_frames)

    # 2. Логика определения ROI
    if args.roi:
        # Если пользователь явно передал --roi в консоли, используем его
        roi = tuple(args.roi)
    else:
        # Если не передал — делегируем менеджеру (он поищет в JSON или спросит)
        roi_manager = ROIManager(config_path="settings/table_config.json")
        roi = roi_manager.get_roi(args.video)

    processor = VideoProcessor(
        video_path=args.video,
        roi=roi,
        monitor=monitor,
        plugins=plugins,
        confidence_threshold=args.confidence,
        output_path=output_video,
        model_name=args.model,
        detection_step=args.step,
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