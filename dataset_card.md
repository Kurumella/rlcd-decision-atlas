---
pretty_name: RLCD Decision Atlas
license: other
language:
- en
task_categories:
- text-classification
- multiple-choice
- question-answering
tags:
- decision-making
- calibration
- multiple-choice
- rlcd
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: test.jsonl
  - split: holdout
    path: holdout.jsonl
---

# rlcd-decision-atlas — a typed-decision dataset

This is a dataset for training and evaluating models that make **decisions** rather than write prose. Every example is one situation, one question, and a short list of declared options — and exactly one of those options is correct. The model's whole job is to put a probability on each option and commit to one. Nothing here asks for free text.

It is assembled from 114 public sources plus one synthetic generator, each converted into the same tiny contract, then split so that a large in-distribution training corpus sits alongside a held-out set of sources the model never trains on — roughly **531k train / 74k validation / 74k test / 27.5k holdout** rows, about 707k typed questions in total.

The name is RLCD — reinforcement learning for calibrated decisions. That is the training method the data was built for, but the data itself is method-agnostic: it is just labelled multiple-choice, and you can train on it with supervised learning, a bandit, or anything else.

## Why this shape

Most classification corpora each speak their own dialect — different label names, different file layouts, a different idea of what an "answer" is. That makes it painful to train one model that decides across domains, and harder still to ask a fair question about calibration, because a model that is confident-and-wrong on banking intents looks nothing like one that is confident-and-wrong on entailment unless you first force both tasks into the same shape.

So everything is forced into the same shape. A sentiment rating, a yes/no reading question, a 77-way intent, a three-way entailment call — all of them become "here is the state, here is the question, pick the letter." Once the tasks share a contract, three things become possible that are awkward otherwise:

- **One model, many decisions.** A single policy can be trained and measured across dozens of task families without per-task heads or bespoke parsing.
- **Calibration you can actually read.** Because the output is a distribution over a small option set, you can compare a model's stated confidence to how often it is right, in the same units, on every source at once. Skewed sources are left skewed on purpose — if hate speech is genuinely rare, a well-calibrated model should say so, and rebalancing the classes would teach it a false prior.
- **A real generalization test.** A handful of sources are quarantined into a holdout split and appear nowhere in training, so you can ask how the model does on decision tasks it has never seen rather than on a fresh sample of ones it has.

## What's in the box

The release is four line-delimited JSON files plus a stats file:

| file | what it holds |
|---|---|
| `train.jsonl` | in-distribution training rows, shuffled across sources |
| `val.jsonl` | in-distribution validation, disjoint from train within each source |
| `test.jsonl` | in-distribution test, disjoint from train and val within each source |
| `holdout.jsonl` | out-of-distribution rows from sources that appear in no other file |
| `stats.json` | per-source and total counts, roles, licenses, and any skipped sources |

`DATA_CARD.md` lists every source with its counts — the exhaustive companion to the summarized tables below.

## The data contract

Every row is a JSON object with a handful of fields:

```jsonc
{
  "primitive": "choice",   // "choice" | "score" | "noul"
  "context":   "...",      // the state to decide on — non-empty, may be truncated
  "question":  "...",      // what is being asked
  "choices":   ["...", "..."],  // 2..26 bare, single-line, unique option strings
  "ordered":   false,
  "answer":    0,          // index of the correct choice in `choices` (or null if unlabelled)
  "source":    "ag_news",  // which source produced the row
  "id":        "ag_news-0" // stable per-row identifier
}
```

There are three primitives, and the difference between them is only how the options relate to each other:

- **`choice`** — an unordered set of options, from 2 up to 26. The answer is one index. Options are shuffled, so the correct answer lands in every position about equally often and the letter carries no signal on its own.
- **`score`** — an *ordered* scale, listed low to high (star ratings, sentiment levels). The order is meaningful, so these options are never shuffled.
- **`noul`** — a yes/no call, rendered as exactly `["true", "false"]` with index 0 meaning the affirmative. Every yes/no recast was checked so that "true" means the thing the question actually asks.

A row becomes a prompt by listing the options as `A) …`, `B) …` and ending on a cue for the answer, so a model can answer by emitting a single letter — which is why options are single-line and there are at most 26 of them. Every row is guaranteed to hold 2–26 options, each non-empty and single-line, all unique after stripping, with an answer index that is either in range or `null`.

### None of the above

Roughly a third of the `choice` and `score` rows carry a synthetic "None of the above" option. Half the time it replaces the true answer and becomes correct; half the time it is an extra distractor. Either way exactly one original option is dropped, so the option count never changes and the presence of the option leaks nothing about whether it is the answer — only the context can tell you. Yes/no rows are never touched.

## Splits

In-distribution rows are split **per source and grouped by context**, so every question that shares a state string lands in the same split — a passage with five questions never has some questions in train and others in test. Validation and test target about 1,000 rows per source each (scaled down for small sources), and train takes the rest up to a cap of 8,000. Across the whole corpus that lands near a 90 / 5 / 5 split.

The holdout is different in kind. Whole sources are set aside — currently MMLU, MedMCQA, ANLI, TruthfulQA, Climate-FEVER, GPQA, and all of the BBQ bias categories — and their rows go only to `holdout.jsonl`, roughly 2,000 per source. Nothing from a holdout source ever reaches train, validation, or test, so the holdout measures decisions on unseen task families rather than unseen examples of seen ones.

Each source is generated from its own deterministic stream keyed by the source name, so the release is reproducible and each source's rows and splits are fixed independently of the others — the mixture is stable rather than a lucky draw.

## The numbers

In this release:

| split | rows | primitives (rows) | "none of the above" | median context |
|---|---:|---|---:|---:|
| train | 531,088 | choice 311,981 · noul 192,440 · score 26,667 | ~32% | 122 chars |
| validation | 73,864 | choice 43,072 · noul 27,459 · score 3,333 | ~31% | 119 chars |
| test | 73,864 | choice 43,072 · noul 27,459 · score 3,333 | ~31% | 120 chars |
| holdout | 27,548 | choice 27,548 | ~50% | 203 chars |

- **114 sources are included** — 98 feed the in-distribution splits and 16 are holdout-only. One further source — GPQA — is gated on the Hub and is not included here; it is recorded as skipped in `stats.json`.
- **Option counts span the full range.** In train: 210k rows are yes/no (K=2), 222k sit at 3–5 options, 44k at 6–10, 45k at 11–20, and 9k use 21–26. Sources drawn from very large label sets (banking intents, CLINC, MASSIVE, language ID) sample a fresh subset of 4–26 real labels per row, so their K varies row to row.
- **Yes/no answers run about 45% true to 55% false** in aggregate, but individual sources keep their natural rates — some are near even, some are heavily one-sided, and that is intentional.
- **Contexts are short by default** — a median of ~120 characters — with passages, tables, and dialogues truncated to 1,500 characters. A small tail of exam questions (a few dozen rows from MMLU-Pro, MMLU, and MedQA) runs longer, up to ~4,700 characters, because the question itself is that long and clipping it would throw away what is being asked.
- **Licenses are recorded, not filtered.** Every source carries its license string in `stats.json` as metadata only. The spread runs from permissive (MIT, Apache-2.0, the various CC-BY variants) through unstated to a few copyleft and non-commercial terms. Treat the license column as a starting point for your own review before redistributing anything — a source's presence here is not clearance to publish it.

## The sources

Grouped by the kind of decision they ask for. The `K` column shows a representative option count; large-label sources sample a subset per row, so their K varies. The rows column is the training count, or the holdout count for holdout-only sources.

#### Commonsense and reasoning

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `social_iqa` | allenai/social_i_qa | choice | 3 | train | CC-BY-4.0 | 8,000 |
| `cosmos_qa` | allenai/cosmos_qa | choice | 4 | train | CC-BY-4.0 | 8,000 |
| `swag` | allenai/swag (regular) | choice | 4 | train | unstated | 8,000 |
| `quartz` | allenai/quartz | choice | 2 | train | CC-BY-4.0 | 1,864 |
| `winogrande` | allenai/winogrande (winogrande_xl) | choice | 2 | train | unstated | 8,000 |
| `logiqa` | lucasmccabe/logiqa | choice | 4 | train | unstated | 6,634 |
| `reclor` | metaeval/reclor | choice | 4 | train | research | 3,138 |
| `codah` | jaredfern/codah (codah) | choice | 4 | train | ODC-BY | 2,219 |
| `hellaswag` | Rowan/hellaswag | choice | 4 | train | MIT | 8,000 |
| `piqa` | ybisk/piqa | choice | 2 | train | AFL-3.0 | 8,000 |

#### Science and knowledge exams

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `mmlu` | cais/mmlu (all) | choice |  | holdout | MIT | 2,000 |
| `mmlu_pro` | TIGER-Lab/MMLU-Pro | choice | 10 | train | MIT | 8,000 |
| `arc` | allenai/ai2_arc (Challenge+Easy) | choice | 4 | train | CC-BY-SA-4.0 | 5,787 |
| `openbookqa` | allenai/openbookqa (main) | choice | 4 | train | Apache-2.0 | 3,954 |
| `commonsense_qa` | tau/commonsense_qa | choice | 5 | train | MIT | 8,000 |
| `sciq` | allenai/sciq | choice | 4 | train | CC-BY-NC-3.0 | 8,000 |
| `qasc` | allenai/qasc | choice | 8 | train | CC-BY-4.0 | 7,060 |
| `medmcqa` | openlifescienceai/medmcqa | choice |  | holdout | Apache-2.0 | 2,000 |
| `medqa` | GBaker/MedQA-USMLE-4-options | choice | 4 | train | unstated | 8,000 |
| `head_qa` | dvilares/head_qa | choice | 5 | train | MIT | 8,000 |
| `gpqa` | Idavidrein/gpqa (gpqa_main) | choice |  | holdout (gated, skipped) | gated/unstated | 0 |

#### Math (multiple choice)

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `aqua_rat` | deepmind/aqua_rat (raw) | choice | 5 | train | Apache-2.0 | 8,000 |
| `math_qa` | allenai/math_qa | choice | 5 | train | Apache-2.0 | 8,000 |

#### Reading comprehension

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `race` | ehovy/race (all) | choice | 4 | train | research | 8,000 |
| `quail` | textmachinelab/quail | choice | 4 | train | CC-BY-NC-SA-4.0 | 8,000 |
| `dream` | dataset-org/dream | choice | 3 | train | unstated | 8,000 |
| `boolq` | google/boolq | noul | 2 | train | CC-BY-SA-3.0 | 8,000 |

#### Natural language inference

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `mnli` | nyu-mll/multi_nli | choice | 3 | train | unstated | 8,000 |
| `snli` | stanfordnlp/snli | choice | 3 | train | CC-BY-SA-4.0 | 8,000 |
| `anli` | facebook/anli | choice |  | holdout | CC-BY-NC-4.0 | 2,000 |
| `scitail` | allenai/scitail (tsv_format) | noul | 2 | train | Apache-2.0 | 8,000 |
| `folio` | tasksource/folio | choice | 3 | train | CC/MIT | 964 |
| `glue_qnli` | nyu-mll/glue (qnli) | noul | 2 | train | permissive | 3,462 |
| `glue_rte` | nyu-mll/glue (rte) | noul | 2 | train | permissive | 223 |
| `superglue_cb` | aps/super_glue (cb) | choice | 3 | train | permissive | 46 |
| `superglue_copa` | aps/super_glue (copa) | choice | 2 | train | permissive | 80 |

#### Fact verification and truthfulness

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `vitaminc` | tals/vitaminc | choice | 3 | train | CC-BY-SA-3.0 | 8,000 |
| `fever` | fever/fever (v1.0) | choice | 3 | train | CC-BY-SA-3.0 | 8,000 |
| `health_fact` | ImperialCollegeLondon/health_fact | choice | 4 | train | MIT | 8,000 |
| `pubmedqa` | qiaojin/PubMedQA (pqa_labeled) | choice | 3 | train | MIT | 800 |
| `tab_fact` | ibm-research/tab_fact (tab_fact) | noul | 2 | train | CC-BY-4.0 | 8,000 |
| `halueval` | pminervini/HaluEval (qa) | noul | 2 | train | unstated | 8,000 |
| `squad_v2` | rajpurkar/squad_v2 | noul | 2 | train | CC-BY-SA-4.0 | 8,000 |
| `climate_fever` | tdiggelm/climate_fever | choice |  | holdout | unstated | 1,535 |
| `truthfulqa` | truthfulqa/truthful_qa (multiple_choice) | choice |  | holdout | Apache-2.0 | 817 |

#### Paraphrase

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `paws` | google-research-datasets/paws (labeled_final) | noul | 2 | train | free-to-use | 8,000 |
| `glue_mrpc` | nyu-mll/glue (mrpc) | noul | 2 | train | permissive | 2,076 |
| `glue_qqp` | nyu-mll/glue (qqp) | noul | 2 | train | permissive | 8,000 |
| `med_q_pairs` | curaihealth/medical_questions_pairs | noul | 2 | train | unstated | 1,048 |

#### Safety, ethics, and bias

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `ethics_commonsense` | hendrycks/ethics (commonsense) | noul | 2 | train | MIT | 8,000 |
| `ethics_justice` | hendrycks/ethics (justice) | noul | 2 | train | MIT | 8,000 |
| `ethics_deontology` | hendrycks/ethics (deontology) | noul | 2 | train | MIT | 8,000 |
| `ethics_virtue` | hendrycks/ethics (virtue) | noul | 2 | train | MIT | 8,000 |
| `moral_stories` | demelin/moral_stories (full) | noul | 2 | train | MIT | 8,000 |
| `ethos_binary` | iamollas/ethos (binary) | noul | 2 | train | AGPL/GPL | 800 |
| `tweets_hate` | tweets-hate-speech-detection/tweets_hate_speech_detection | noul | 2 | train | GPL-3.0 | 8,000 |

#### Sentiment, emotion, and rating

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `sst5` | SetFit/sst5 | score | 5 | train | unstated | 8,000 |
| `yelp` | Yelp/yelp_review_full | score | 5 | train | unstated | 8,000 |
| `imdb` | stanfordnlp/imdb | noul | 2 | train | permissive | 8,000 |
| `rotten_tomatoes` | cornell-movie-review-data/rotten_tomatoes | noul | 2 | train | unstated | 8,000 |
| `amazon_polarity` | fancyzhx/amazon_polarity | noul | 2 | train | Apache-2.0 | 8,000 |
| `app_reviews` | sealuzh/app_reviews | score | 5 | train | unstated | 8,000 |
| `emotion` | dair-ai/emotion (split) | choice | 6 | train | unstated | 8,000 |
| `financial_phrasebank` | takala/financial_phrasebank (sentences_allagree) | choice | 3 | train | CC-BY-NC-SA-3.0 | 1,812 |

#### Topic and intent

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `ag_news` | fancyzhx/ag_news | choice | 4 | train | unstated | 8,000 |
| `dbpedia_14` | fancyzhx/dbpedia_14 | choice | 14 | train | CC-BY-SA-3.0 | 8,000 |
| `yahoo_topics` | community-datasets/yahoo_answers_topics | choice | 10 | train | unstated | 8,000 |
| `newsgroups` | SetFit/20_newsgroups | choice | 20 | train | unstated | 8,000 |
| `trec_coarse` | CogComp/trec (coarse_label) | choice | 6 | train | unstated | 3,952 |
| `trec_fine` | CogComp/trec (fine_label) | choice | 5 | train | unstated | 3,952 |
| `clinc_oos` | clinc/clinc_oos (plus) | choice | 4 | train | CC-BY-3.0 | 8,000 |
| `massive_intent` | AmazonScience/massive (en-US) | choice | 18 | train | CC-BY-4.0 | 8,000 |
| `massive_scenario` | AmazonScience/massive (en-US) | choice | 18 | train | CC-BY-4.0 | 8,000 |
| `subj` | SetFit/subj | noul | 2 | train | unstated | 8,000 |
| `sms_spam` | ucirvine/sms_spam | noul | 2 | train | unstated | 3,574 |
| `lang_id` | papluca/language-identification | choice | 12 | train | derived | 8,000 |
| `banking77` | legacy-datasets/banking77 | choice | 5 | train | CC-BY-4.0 | 8,000 |
| `bitext` | bitext/Bitext-customer-support-llm-chatbot-training-dataset | choice | 9 | train | unstated | 8,000 |

#### Synthetic decisions

| source | Hugging Face id | type | K | split role | license | rows |
|---|---|---|---|---|---|---|
| `triage` | (synthetic) | choice | 4 | train | synthetic | 8,000 |

#### Config bundles (each config is its own source; see DATA_CARD.md for the full list)

| family | Hugging Face id | type | split role | configs | rows |
|---|---|---|---|---|---|
| `tweet_eval_*` | cardiffnlp/tweet_eval | choice / noul | train | 5 | 29,653 |
| `babi_nli_*` | tasksource/babi_nli | noul | train | 20 | 23,990 |
| `bbq_*` | heegyu/bbq | choice | holdout | 11 | 19,196 |

## Loading it

Each line of every file is one JSON object, so read it with whatever you already use:

```python
import json

with open("train.jsonl") as f:
    rows = [json.loads(line) for line in f]

q = rows[0]
print(q["source"], q["primitive"], q["choices"], q["answer"])
```

or with the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("goutam/rlcd-decision-atlas")   # train, validation, test, holdout
```

A row turns into a prompt by listing the options as lettered lines and ending on a cue for the answer letter — one natural rendering is:

```
User: Context:
<context>

Question: <question>
Options:
A) <choice 0>
B) <choice 1>
...
Answer with the letter only.
Assistant: The answer is
```

The `holdout` split has the same schema as the others and is meant to be read exactly like `test` — it is simply a stricter test, since none of its sources appear in the other splits.

## Good to know

- **No invented answers.** A source is only included if it has a real fixed option set or can be recast as a clean yes/no. Distractors are never fabricated — the only sampling that happens is drawing a subset from a source's genuine label set when it has more than 26 labels.
- **Positions carry no signal.** Unordered options are permuted per row, so a model cannot learn "the answer is usually B." Ordered scales and yes/no rows keep their fixed order because the order is the point.
- **Natural class balance is preserved.** Skewed sources stay skewed. For a model meant to be calibrated, the real base rate is information, not noise.
- **Reproducible and auditable.** The release is built deterministically, and `stats.json` records exactly what went in — per-source counts, split roles, licenses, and any source that was skipped.
- **Licenses are metadata, not clearance.** Each source's license string is recorded for reference; check it yourself before redistributing any subset.
