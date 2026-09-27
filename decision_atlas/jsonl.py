"""Reading and writing Decisions as JSON lines. Both directions validate every row, so a bad
row surfaces immediately rather than silently landing in a file."""
from __future__ import annotations

from pathlib import Path

from decision_atlas.decision import Decision


def read_decisions(path) -> list[Decision]:
    rows: list[Decision] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(Decision.from_json(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows


def write_decisions(path, rows: list[Decision]) -> None:
    # Validate the whole batch before touching the file, so an invalid row leaves nothing behind.
    lines = [r.check().to_json() for r in rows]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in lines)
