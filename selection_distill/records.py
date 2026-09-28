"""Small UTF-8 record readers and atomic writers for experiment artifacts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source.name}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source.name}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source.name}:{exc.lineno}: invalid JSON: {exc.msg}") from exc


def _atomic_write(path: str | Path, write_content) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            write_content(handle)
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    def write_content(stream) -> None:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each JSONL row must be an object")
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    _atomic_write(path, write_content)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{source.name}: CSV header is required")
        return [dict(row) for row in reader]


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    def write_content(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _atomic_write(path, write_content)


def write_json(path: str | Path, value: Any) -> None:
    def write_content(stream) -> None:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    _atomic_write(path, write_content)
