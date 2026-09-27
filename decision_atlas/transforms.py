"""Transforms applied while assembling splits: sampling a subset of options, adding a
none-of-the-above option, permuting unordered options, cleaning label names, dropping blank
contexts, and splitting a source so a shared context never crosses splits.

Every source of randomness is a caller-supplied `random.Random`, so a build is reproducible.
"""
from __future__ import annotations

import random

from decision_atlas.decision import NONE_OPTION, Decision


def sample_options(pool: list[str], truth: str, k: int, rng: random.Random) -> tuple[list[str], int]:
    """Return `truth` plus up to k-1 other members of `pool`, with `truth` inserted at a random
    position. Used when a source has more real labels than fit in a single question."""
    others = [o for o in pool if o != truth]
    rng.shuffle(others)
    kept = others[: max(k - 1, 0)]
    slot = rng.randrange(len(kept) + 1)
    kept.insert(slot, truth)
    return kept, slot


def add_none_option(d: Decision, rng: random.Random,
                    p_correct: float = 0.15, p_distractor: float = 0.35) -> Decision:
    """Sometimes fold a "none of the above" option in. With probability `p_correct` it replaces
    the true option and becomes the answer; with `p_distractor` it replaces a wrong option and is
    just a distractor; otherwise the row is returned unchanged. Either branch removes exactly one
    original option, so the option count is constant and the extra option leaks nothing about
    whether it is correct. Never applied to yes/no rows."""
    if d.primitive == "noul" or NONE_OPTION in d.choices:
        return d
    roll = rng.random()
    if roll < p_correct:
        drop = d.answer
    elif roll < p_correct + p_distractor:
        drop = rng.choice([i for i in range(d.n_options) if i != d.answer])
    else:
        return d
    kept = [o for i, o in enumerate(d.choices) if i != drop] + [NONE_OPTION]
    if drop == d.answer:
        gold = len(kept) - 1
    else:
        gold = d.answer - (1 if drop < d.answer else 0)
    return Decision(d.primitive, d.context, d.question, kept, d.ordered, gold, d.source, d.id)


def permute_options(d: Decision, rng: random.Random) -> Decision:
    """Shuffle the options of an unordered `choice` row so answer position carries no signal.
    A none-of-the-above option is pinned last. Ordered scales and yes/no rows are left alone."""
    if d.primitive != "choice" or d.ordered:
        return d
    order = [i for i, o in enumerate(d.choices) if o != NONE_OPTION]
    rng.shuffle(order)
    order += [i for i, o in enumerate(d.choices) if o == NONE_OPTION]
    reordered = [d.choices[i] for i in order]
    gold = None if d.answer is None else order.index(d.answer)
    return Decision(d.primitive, d.context, d.question, reordered, d.ordered, gold, d.source, d.id)


def clean_label_names(names: list[str], source: str) -> list[str]:
    """Strip and single-line a source's label names so they are valid options. Two names that
    collapse to the same string would be indistinguishable as options, so that is an error."""
    cleaned = [" ".join(str(n).splitlines()).strip() for n in names]
    if any(not n for n in cleaned):
        raise ValueError(f"{source}: empty label name in {names!r}")
    collisions = sorted({n for n in cleaned if cleaned.count(n) > 1})
    if collisions:
        raise ValueError(f"{source}: duplicate label names after cleaning: {collisions!r}")
    return cleaned


def drop_blank_contexts(rows: list[Decision]) -> tuple[list[Decision], int]:
    """Remove rows whose context is empty or whitespace-only; report how many were removed."""
    kept = [r for r in rows if r.context.strip()]
    return kept, len(rows) - len(kept)


def split_by_context(rows: list[Decision], per_source: int, n_val: int, n_test: int,
                     rng: random.Random) -> tuple[list, list, list]:
    """Partition rows so every row sharing a context lands in one split. Context groups are
    shuffled, then test is filled to at least n_test, then val, then train up to per_source; the
    tail overshoot is dropped rather than moved, so no context ever straddles two splits."""
    grouped: dict[str, list[Decision]] = {}
    for r in rows:
        grouped.setdefault(r.context, []).append(r)
    groups = list(grouped.values())
    rng.shuffle(groups)
    train: list[Decision] = []
    val: list[Decision] = []
    test: list[Decision] = []
    for group in groups:
        if len(test) < n_test:
            test += group
        elif len(val) < n_val:
            val += group
        elif len(train) < per_source:
            train += group
        else:
            break
    return train[:per_source], val[:n_val], test[:n_test]
