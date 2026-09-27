"""Render a Decision into the prompt a model answers with a single option letter. This is one
convention for consuming the data — the rows themselves carry no prompt text."""
from __future__ import annotations

from decision_atlas.decision import OPTION_LETTERS, Decision


def to_prompt(d: Decision) -> str:
    header = f"User: Context:\n{d.context.strip()}\n\nQuestion: {d.question.strip()}\nOptions:"
    options = [f"{letter}) {choice}" for letter, choice in zip(OPTION_LETTERS, d.choices)]
    footer = "Answer with the letter only.\nAssistant: The answer is"
    return "\n".join([header, *options, footer])
