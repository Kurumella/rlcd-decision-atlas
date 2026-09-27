"""Assemble the train/validation/test/holdout splits from the registered sources and write them.

Each source is drawn from its own deterministic stream (keyed by name), so its rows and splits are
fixed independently of which other sources are built. Holdout sources are quarantined into their
own file; every other source is split so a shared context never crosses splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from decision_atlas.jsonl import write_decisions
from decision_atlas.sources import CAPS, HOLDOUT_SOURCES, SOURCE_LICENSE, SOURCES
from decision_atlas.transforms import (
    add_none_option,
    drop_blank_contexts,
    permute_options,
    split_by_context,
)

HOLDOUT_PER_SOURCE = 2000


def source_rng(seed: int, name: str) -> random.Random:
    """A Random keyed by (seed, source name) so each source is reproducible on its own."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8]
    return random.Random(int.from_bytes(digest, "big"))


def _finalize(rows, rng):
    # Per row: maybe add none-of-the-above, then permute the options. Order matters for the stream.
    return [permute_options(add_none_option(r, rng), rng) for r in rows]


def build(out_dir: str, per_source: int = 8000, n_val: int = 1000, n_test: int = 1000,
          seed: int = 0, sources: list[str] | None = None,
          holdout_per_source: int = HOLDOUT_PER_SOURCE) -> dict:
    train, val, test, holdout = [], [], [], []
    stats: dict = {}
    dropped: dict = {}
    for name in sources or list(SOURCES):
        lic = SOURCE_LICENSE.get(name, "unstated")
        rng = source_rng(seed, name)
        try:
            rows, dropped[name] = drop_blank_contexts(SOURCES[name](rng))
        except Exception as exc:  # noqa: BLE001 -- a gated or broken source is logged, not fatal
            stats[name] = {"error": f"{type(exc).__name__}: {exc}", "license": lic}
            continue
        if name in HOLDOUT_SOURCES:
            rng.shuffle(rows)
            held = _finalize(rows[:holdout_per_source], rng)
            holdout += held
            stats[name] = {"holdout": len(held), "role": "holdout", "license": lic,
                           "primitive": (held[0].primitive if held else None)}
            continue
        cap = CAPS.get(name, per_source)
        nv = min(n_val, max(len(rows) // 10, 1)) if len(rows) < 3000 else n_val
        nt = min(n_test, max(len(rows) // 10, 1)) if len(rows) < 3000 else n_test
        tr, va, te = split_by_context(rows, cap, nv, nt, rng)
        tr, va, te = _finalize(tr, rng), _finalize(va, rng), _finalize(te, rng)
        train += tr
        val += va
        test += te
        stats[name] = {"train": len(tr), "val": len(va), "test": len(te), "role": "train",
                       "license": lic, "primitive": (tr[0].primitive if tr else None),
                       "k": (tr[0].n_options if tr else None)}
    random.Random(seed).shuffle(train)  # order the merged train file; per-source content unaffected
    out = Path(out_dir)
    write_decisions(out / "train.jsonl", train)
    write_decisions(out / "val.jsonl", val)
    write_decisions(out / "test.jsonl", test)
    write_decisions(out / "holdout.jsonl", holdout)
    stats["total"] = {"train": len(train), "val": len(val), "test": len(test),
                      "holdout": len(holdout)}
    stats["skipped"] = dropped
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="decision_atlas")
    commands = parser.add_subparsers(dest="cmd", required=True)
    b = commands.add_parser("build", help="build the dataset splits")
    b.add_argument("--out", default="data")
    b.add_argument("--per-source", type=int, default=8000)
    b.add_argument("--n-val", type=int, default=1000)
    b.add_argument("--n-test", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sources", nargs="*", default=None)
    b.add_argument("--holdout-per-source", type=int, default=HOLDOUT_PER_SOURCE)
    args = parser.parse_args()
    result = build(args.out, args.per_source, args.n_val, args.n_test, args.seed,
                   args.sources, args.holdout_per_source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
