import pytest

from decision_atlas.decision import MAX_OPTIONS, NONE_OPTION, Decision
from decision_atlas.jsonl import read_decisions, write_decisions
from decision_atlas.prompt import to_prompt


def make(**overrides) -> Decision:
    fields = dict(primitive="choice", context="ctx", question="Which?",
                  choices=["a", "b", "c"], answer=1)
    fields.update(overrides)
    return Decision(**fields)


def test_n_options_counts_choices():
    assert make().n_options == 3


def test_check_accepts_each_primitive():
    make().check()
    make(primitive="noul", choices=["true", "false"], answer=0).check()
    make(primitive="score", ordered=True, choices=["low", "high"], answer=0).check()


def test_check_rejects_unknown_primitive():
    with pytest.raises(ValueError, match="primitive"):
        make(primitive="ranking").check()


def test_check_enforces_option_count():
    with pytest.raises(ValueError, match="need 2"):
        make(choices=["solo"], answer=0).check()
    with pytest.raises(ValueError, match="need 2"):
        make(choices=[str(i) for i in range(MAX_OPTIONS + 1)], answer=0).check()


def test_check_rejects_blank_or_multiline_options():
    with pytest.raises(ValueError, match="non-empty single-line"):
        make(choices=["a", "  ", "b"]).check()
    with pytest.raises(ValueError, match="non-empty single-line"):
        make(choices=["a", "b\nc", "d"]).check()


def test_check_rejects_duplicate_options():
    with pytest.raises(ValueError, match="duplicate"):
        make(choices=["a", "a ", "b"]).check()


def test_check_noul_must_be_true_false():
    with pytest.raises(ValueError, match="noul"):
        make(primitive="noul", choices=["yes", "no"], answer=0).check()


def test_check_score_requires_ordered():
    with pytest.raises(ValueError, match="ordered"):
        make(primitive="score", ordered=False).check()


def test_check_answer_range_and_type():
    with pytest.raises(ValueError, match="answer"):
        make(answer=9).check()
    with pytest.raises(ValueError, match="answer"):
        make(answer=1.0).check()
    assert make(answer=None).check().answer is None


def test_json_round_trip():
    d = make(source="unit", id="x1")
    assert Decision.from_json(d.to_json()) == d


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [make(id="1"), make(id="2", choices=["x", "y"], answer=0)]
    write_decisions(path, rows)
    assert read_decisions(path) == rows


def test_writer_validates_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "d.jsonl"
    with pytest.raises(ValueError, match="answer"):
        write_decisions(path, [make(id="ok"), make(id="bad", answer=9)])
    assert not path.exists()


def test_reader_reports_offending_line(tmp_path):
    path = tmp_path / "d.jsonl"
    good = make(id="1").to_json()
    corrupt = good.replace('"answer": 1', '"answer": 9')
    path.write_text(good + "\n" + corrupt + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"d\.jsonl:2:"):
        read_decisions(path)


def test_to_prompt_layout():
    text = to_prompt(make(context="  a ticket  ", question="Where?").check())
    assert text.startswith("User: Context:\na ticket\n\nQuestion: Where?\nOptions:\nA) a\n")
    assert text.endswith("Answer with the letter only.\nAssistant: The answer is")


def test_module_constants():
    assert NONE_OPTION == "None of the above"
    assert MAX_OPTIONS == 26
