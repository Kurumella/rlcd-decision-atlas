import random
from collections import Counter

import pytest

from decision_atlas.decision import NONE_OPTION, Decision
from decision_atlas.synthetic import make_triage
from decision_atlas.transforms import (
    add_none_option,
    clean_label_names,
    drop_blank_contexts,
    permute_options,
    sample_options,
    split_by_context,
)


def make(**overrides) -> Decision:
    fields = dict(primitive="choice", context="c", question="q", choices=["x", "y", "z"], answer=1)
    fields.update(overrides)
    return Decision(**fields).check()


def test_sample_options_holds_truth_once():
    pool = [f"label_{i}" for i in range(50)]
    opts, gold = sample_options(pool, "label_7", 6, random.Random(0))
    assert len(opts) == 6 and len(set(opts)) == 6
    assert opts[gold] == "label_7" and opts.count("label_7") == 1


def test_sample_options_small_pool_keeps_all():
    opts, gold = sample_options(["a", "b"], "b", 6, random.Random(1))
    assert sorted(opts) == ["a", "b"] and opts[gold] == "b"


def test_add_none_option_mix_and_constant_count():
    rng = random.Random(0)
    seen = Counter()
    for _ in range(4000):
        out = add_none_option(make(), rng)
        assert out.n_options == 3
        if NONE_OPTION not in out.choices:
            seen["absent"] += 1
        elif out.choices[out.answer] == NONE_OPTION:
            seen["correct"] += 1
        else:
            seen["distractor"] += 1
            assert out.choices[-1] == NONE_OPTION
    assert 0.10 < seen["correct"] / 4000 < 0.20
    assert 0.30 < seen["distractor"] / 4000 < 0.40


def test_add_none_option_never_touches_noul():
    q = make(primitive="noul", choices=["true", "false"], answer=0)
    for _ in range(50):
        assert add_none_option(q, random.Random(1)) == q


def test_permute_spreads_positions_and_preserves_answer():
    rng = random.Random(0)
    positions = Counter()
    for _ in range(2000):
        out = permute_options(make(choices=["a", "b", "c", "d"], answer=2), rng).check()
        assert out.choices[out.answer] == "c"
        positions[out.answer] += 1
    assert all(positions[i] / 2000 > 0.15 for i in range(4))


def test_permute_pins_none_last():
    rng = random.Random(0)
    for _ in range(200):
        out = permute_options(make(choices=["a", "b", NONE_OPTION], answer=0), rng).check()
        assert out.choices[-1] == NONE_OPTION


def test_permute_leaves_ordered_and_noul_alone():
    rng = random.Random(0)
    score = make(primitive="score", ordered=True, choices=["low", "mid", "high"], answer=0)
    noul = make(primitive="noul", choices=["true", "false"], answer=1)
    for _ in range(20):
        assert permute_options(score, rng) == score
        assert permute_options(noul, rng) == noul


def test_clean_label_names_strips_flattens_and_guards():
    assert clean_label_names([" a ", "two\nlines"], "src") == ["a", "two lines"]
    with pytest.raises(ValueError, match="duplicate"):
        clean_label_names(["x", " x "], "src")
    with pytest.raises(ValueError, match="empty"):
        clean_label_names(["x", "  "], "src")


def test_drop_blank_contexts_counts_removed():
    kept, removed = drop_blank_contexts([make(id="a"), make(id="b", context="  \n ")])
    assert [d.id for d in kept] == ["a"] and removed == 1


def test_split_by_context_sizes_and_disjoint():
    rows = [make(id=f"{c}-{j}", context=f"ctx {c}") for c in range(40) for j in range(3)]
    tr, va, te = split_by_context(rows, per_source=60, n_val=20, n_test=20, rng=random.Random(0))
    assert len(tr) == 60 and len(va) == 20 and len(te) == 20
    tr_c, va_c, te_c = ({d.context for d in part} for part in (tr, va, te))
    assert not (tr_c & va_c) and not (tr_c & te_c) and not (va_c & te_c)


def test_triage_shape_and_determinism():
    rows = make_triage(10, random.Random(0))
    assert len(rows) == 30
    assert Counter(d.primitive for d in rows) == {"choice": 10, "score": 10, "noul": 10}
    for d in rows:
        d.check()
        assert d.source == "triage"
    assert make_triage(5, random.Random(7)) == make_triage(5, random.Random(7))
