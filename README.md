# rlcd-decision-atlas

Data preparation for **rlcd-decision-atlas** — a typed-decision dataset that folds many public sources into one contract: a situation, a question, and 2–26 declared options with exactly one correct answer. It is built for training and evaluating models that *decide* (and stay calibrated while deciding) rather than generate text.

This repository is the recipe, not the data. It downloads public datasets, converts each into the shared contract, injects a "none of the above" option on some rows, shuffles unordered options, and splits everything into train/validation/test plus an out-of-distribution holdout. The built dataset is published separately at **[goutam/rlcd-decision-atlas](https://huggingface.co/datasets/goutam/rlcd-decision-atlas)**.

- `dataset_card.md` — the card published with the dataset (what it is, how to load it, the full stats).
- `DATA_CARD.md` — the per-source table, regenerated from a build's `stats.json`.

## Why it matters

Classification datasets each speak their own dialect — different label names, different file layouts, different ideas of what an "answer" even is — which makes it hard to train, or fairly evaluate, a single model that decides across domains. rlcd-decision-atlas removes that friction: every one of 100+ sources becomes the same shape — a context, a question, and a short list of options with exactly one correct answer — so a model can learn to make **calibrated decisions** across commonsense, exams, entailment, fact-checking, ethics, sentiment, and intent at once, and be measured in the same units everywhere.

Three choices make it genuinely useful rather than just large:

- **One contract, many domains.** A single policy can be trained and scored across dozens of task families without per-task heads or bespoke parsing.
- **Calibration you can read.** The output is a distribution over a small option set, so stated confidence can be compared to how often the model is right — on every source at once. Natural class balance is left intact on purpose, because the real base rate is information a calibrated model should learn, not noise to be scrubbed out.
- **A real generalization test.** Whole sources are quarantined into an out-of-distribution holdout and appear nowhere in training, so you can ask how a model does on decision tasks it has never seen — not just on fresh examples of ones it has.

## Reproduce the dataset

```bash
git clone https://github.com/Kurumella/rlcd-decision-atlas.git
cd rlcd-decision-atlas
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install "datasets>=3.2,<4"
python -m decision_atlas build --out data --per-source 8000 --seed 0
```

Python 3.12 is required, and `datasets` must stay on the 3.x line — version 4 and later drop the loading scripts several sources still ship. The build writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `holdout.jsonl`, and `stats.json` into `data/`; the first run pulls a fair amount from the Hugging Face hub and caches it. Most sources are public; GPQA is gated and is skipped cleanly unless you set an `HF_TOKEN`. The build is deterministic — the same version and seed produce the same bytes.

Useful flags:

- `--sources ag_news snli mmlu` — build only the named sources. Each source is deterministic on its own, so a subset reproduces exactly the rows a full build would give it.
- `--per-source 4000` — lower the training cap per source; `--n-val` / `--n-test` size the other splits.
- `--holdout-per-source 2000` — rows contributed by each holdout source.

> **Security note.** The converters call `datasets` with `trust_remote_code=True`, because a few of the pinned public sources still ship a loading script that datasets 3.x will only run with that flag. This executes dataset code from those specific, named Hugging Face repositories at build time — review them if that matters in your environment. The flag is ignored for the parquet-native sources.

## How it works

The `decision_atlas` package keeps each concern in its own module:

- `decision.py` — the `Decision` record and its `check()` validation. A row is `primitive` (`choice` / ordered `score` / yes-no `noul`), `context`, `question`, `choices` (2–26 bare single-line strings), `answer` (an index), plus `source` and `id`.
- `jsonl.py` — `read_decisions` / `write_decisions`, both of which validate every row, so a bad row surfaces loudly rather than landing on disk.
- `fields.py` — pure row-construction helpers (`oneline`, `clip`, `letter_index`, `key_index`, `text_index`, `parse_packed`, `usable`).
- `transforms.py` — the build-time transforms (`sample_options`, `add_none_option`, `permute_options`, `clean_label_names`, `drop_blank_contexts`, `split_by_context`).
- `synthetic.py` — `make_triage`, the one non-public source.
- `sources.py` — one converter per source plus the registry (`SOURCES`, `HOLDOUT_SOURCES`, `CAPS`, `SOURCE_LICENSE`).
- `build.py` — `build()` and the CLI. Each converter returns `Decision`s with a fixed option set and gold index; `build()` drops empty contexts, splits, adds "none of the above", permutes options, and writes the files.

A few things the build guarantees:

- **Positions carry no signal.** Unordered `choice` options are permuted per row; ordered scales and yes/no rows keep their fixed order.
- **No invented answers.** A source is only admitted if it has a real fixed option set or a clean yes/no recast. The one augmentation is "none of the above", which is a neutral catch-all, not a fabricated distractor. Sources with more than 26 labels sample a subset of their *real* labels per row.
- **Split isolation.** In-distribution rows are split per source and grouped by context, so questions sharing a state never straddle splits. Holdout sources appear only in `holdout.jsonl`.
- **Reproducibility.** Each source draws from its own deterministic stream keyed by name, so a source's rows and splits are fixed independently of the others — adding or changing one source does not perturb the rest.

## Adding a source

1. Write `def <name>(rng: random.Random) -> list[Decision]:` in `decision_atlas/sources.py`, using `_load(...)` for the dataset and the `fields`/`transforms` helpers to shape rows. Return `Decision`s with a fixed option set and gold index — do not permute options, add none-of-the-above, or split yourself; `build()` applies those.
2. Register it in `SOURCES`, record its license in `SOURCE_LICENSE`, and — if it should be out-of-distribution — add it to `HOLDOUT_SOURCES`. Cap giant sources via `CAPS`.
3. Smoke-test it against real data: `pytest -m network -k "test_each_source_first_rows_valid and <name>"`.

## Testing

```bash
pip install pytest ruff
ruff check decision_atlas tests
pytest                      # unit tests only (network tests are deselected by default)
pytest -m network -k test_each_source_first_rows_valid   # validate each source against the hub
```

The unit tests cover the record contract and the transforms offline. The `network` tests hit the hub and are opt-in.

## Regenerating the data card

```bash
python scripts/make_data_card.py data/stats.json DATA_CARD.md
```

## License

Apache-2.0 — Copyright 2026 Kurumella. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The datasets this recipe downloads keep their own terms and are not redistributed here; each source's license is recorded in `stats.json` as metadata, and you should review it before redistributing any built subset.
