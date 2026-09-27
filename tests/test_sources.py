import random

import pytest

from decision_atlas.fields import key_index, letter_index, parse_packed, text_index
from decision_atlas.sources import HOLDOUT_SOURCES, SOURCES


def test_field_helpers():
    assert letter_index("A") == 0 and letter_index("c") == 2
    assert key_index(["A", "B", "C"], "B") == 1
    assert text_index(["yes", "no"], "no") == 1
    assert parse_packed("a ) 12 , b ) 34 , c ) 56 , d ) 78") == \
        {"a": "12", "b": "34", "c": "56", "d": "78"}


def test_holdout_sources_are_registered():
    for name in HOLDOUT_SOURCES:
        assert name in SOURCES, name


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_each_source_first_rows_valid(name):
    rows = SOURCES[name](random.Random(0))
    assert len(rows) >= 2
    for d in rows[:200]:
        d.check()


@pytest.mark.network
def test_build_is_split_disjoint_and_isolates_holdout(tmp_path):
    from decision_atlas.build import build
    from decision_atlas.jsonl import read_decisions
    build(str(tmp_path), per_source=200, n_val=50, n_test=50, seed=0,
          sources=["ag_news", "snli", "mmlu"])
    train = {d.context for d in read_decisions(tmp_path / "train.jsonl")}
    val = {d.context for d in read_decisions(tmp_path / "val.jsonl")}
    test = {d.context for d in read_decisions(tmp_path / "test.jsonl")}
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    holdout = read_decisions(tmp_path / "holdout.jsonl")
    assert {d.source for d in holdout} <= HOLDOUT_SOURCES
    assert all(d.source != "mmlu" for d in read_decisions(tmp_path / "train.jsonl"))
