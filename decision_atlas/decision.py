"""The Decision record — the one row type every source is folded into.

A Decision is a situation (`context`), a `question`, and a small list of `choices` with exactly
one gold `answer` index. `primitive` says how the choices relate: an unordered `choice` set, an
ordered `score` scale, or a two-value `noul` yes/no. Everything downstream speaks in Decisions.
"""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass

OPTION_LETTERS: list[str] = list(string.ascii_uppercase)
MAX_OPTIONS = 26
NONE_OPTION = "None of the above"
KINDS = ("choice", "score", "noul")


@dataclass
class Decision:
    primitive: str
    context: str
    question: str
    choices: list[str]
    ordered: bool = False
    answer: int | None = None
    source: str = ""
    id: str = ""

    @property
    def n_options(self) -> int:
        return len(self.choices)

    def check(self) -> "Decision":
        """Raise if this row would be invalid on disk; return self otherwise."""
        kind = self.primitive
        if kind not in KINDS:
            raise ValueError(f"unknown primitive {kind!r}")
        opts = self.choices
        if not isinstance(opts, list) or not all(isinstance(o, str) for o in opts):
            raise ValueError("choices must be a list of str")
        count = len(opts)
        if not 2 <= count <= MAX_OPTIONS:
            raise ValueError(f"need 2..{MAX_OPTIONS} choices, got {count}")
        if any((not o.strip()) or "\n" in o or "\r" in o for o in opts):
            raise ValueError("choices must be non-empty single-line strings")
        if len({o.strip() for o in opts}) != count:
            raise ValueError("duplicate choices")
        if kind == "noul" and opts != ["true", "false"]:
            raise ValueError('noul choices must be exactly ["true", "false"]')
        if kind == "score" and not self.ordered:
            raise ValueError("score questions must set ordered=True")
        if self.answer is not None:
            if type(self.answer) is not int:
                raise ValueError(f"answer must be an int or None, got {self.answer!r}")
            if not 0 <= self.answer < count:
                raise ValueError(f"answer {self.answer} out of range for {count} choices")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Decision":
        return cls(**json.loads(line)).check()
