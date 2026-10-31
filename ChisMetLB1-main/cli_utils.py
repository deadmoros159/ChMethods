"""Небольшая общая функция запуска лабораторных из JSON-файла."""
import json
import sys
from pathlib import Path
from typing import Callable


def run_cli(run_function: Callable[[dict], str], lab_name: str) -> None:
    """Читает путь к JSON из аргумента командной строки и выводит отчёт."""
    if len(sys.argv) != 2:
        script = Path(sys.argv[0]).name
        print(f"Использование: python3 {script} data_lbN.json")
        raise SystemExit(2)

    data_path = Path(sys.argv[1])
    try:
        config = json.loads(data_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Ошибка: файл не найден: {data_path}", file=sys.stderr)
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(
            f"Ошибка JSON в файле {data_path}, строка {exc.lineno}, "
            f"столбец {exc.colno}: {exc.msg}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except OSError as exc:
        print(f"Не удалось прочитать файл {data_path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        print(run_function(config))
    except Exception as exc:
        print(f"{lab_name}: ошибка вычисления: {exc}", file=sys.stderr)
        raise SystemExit(1)
