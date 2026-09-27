"""Row-construction helpers used by the source converters: text hygiene, gold-answer mapping,
and packed-option parsing. These are pure and carry no randomness."""
from __future__ import annotations

import re


def oneline(text) -> str:
    """Collapse all runs of whitespace (including newlines) to single spaces."""
    return " ".join(str(text).split())


def clip(text, limit: int = 1500) -> str:
    """Cap a context at `limit` characters so passages and tables do not blow up the prompt."""
    return str(text)[:limit]


def letter_index(key) -> int:
    """Map a letter answer to a position: 'A'/'a' -> 0, 'B' -> 1, and so on."""
    return ord(str(key).strip().upper()) - ord("A")


def key_index(keys, key) -> int:
    """Position of `key` inside a list of option keys (for answerKey-style gold labels)."""
    return [str(k).strip() for k in keys].index(str(key).strip())


def text_index(options, target) -> int:
    """Position of the option whose text equals the gold answer text."""
    return [o.strip() for o in options].index(str(target).strip())


_PACKED = re.compile(r"([a-e])\s*\)\s*(.*?)(?=\s*[a-e]\s*\)|$)", re.DOTALL)


def parse_packed(packed: str) -> dict:
    """Split a single 'a ) 1 , b ) 2 , ...' string into {'a': '1', 'b': '2', ...}."""
    return {m.group(1): " ".join(m.group(2).split()).strip(" ,") for m in _PACKED.finditer(packed)}


def usable(options: list[str]) -> bool:
    """True when an option list is admissible: at least two options, each non-empty and
    single-line, and all distinct after stripping. An empty or duplicated option would be
    rejected when the row is written, so rows that fail this are dropped at conversion time."""
    if len(options) < 2:
        return False
    if any((not o.strip()) or "\n" in o or "\r" in o for o in options):
        return False
    return len({o.strip() for o in options}) == len(options)
