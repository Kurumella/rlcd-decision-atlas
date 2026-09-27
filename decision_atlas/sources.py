"""Source converters and the registry.

Each converter is `name(rng) -> list[Decision]`: it pulls one public dataset off the Hugging Face
hub and maps every usable row into a Decision with a fixed option set and a gold index. Converters
do not shuffle options, add none-of-the-above, or split — that is the build's job. The registry at
the bottom names every source, records its license, marks the out-of-distribution holdout members,
and expands the config bundles (tweet_eval, BBQ, bAbI-NLI).
"""
from __future__ import annotations

import re

from decision_atlas.decision import MAX_OPTIONS, Decision
from decision_atlas.fields import (
    clip,
    key_index,
    letter_index,
    oneline,
    parse_packed,
    text_index,
    usable,
)
from decision_atlas.synthetic import make_triage
from decision_atlas.transforms import clean_label_names, sample_options


def _patch_list_feature() -> None:
    # The hub is re-exporting parquet with the datasets>=4 `List` feature type, which datasets 3.x
    # cannot deserialize. Its `Sequence` has the same dict-of-lists semantics the converters read
    # (e.g. row["choices"]["text"]), so register it under the missing name.
    import datasets
    from datasets.features import features as feat
    if "List" not in feat._FEATURE_TYPES:
        feat._FEATURE_TYPES["List"] = datasets.Sequence


def _load(path: str, split: str, **kw):
    # `path` is the repo id; a config goes through `name=`. Several older sources still ship a
    # loading script that datasets 3.x will only run with trust_remote_code=True (and which
    # datasets>=4 refuses entirely); the flag is harmless for parquet-native sources.
    from datasets import load_dataset
    _patch_list_feature()
    kw.setdefault("trust_remote_code", True)
    return load_dataset(path, split=split, **kw)


# ---- seed sources -----------------------------------------------------------------------------

def bitext(rng):
    ds = _load("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "train")
    raw = sorted(set(ds["intent"]))
    lookup = dict(zip(raw, clean_label_names(raw, "bitext")))
    labels = list(lookup.values())
    rows = []
    for i, row in enumerate(ds):
        opts, gold = sample_options(labels, lookup[row["intent"]], rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", row["instruction"], "What is the customer's intent?",
                             opts, False, gold, "bitext", f"bitext-{i}"))
    return rows


def banking77(rng):
    # legacy-datasets/banking77 is the hub's parquet mirror; the PolyAI original is script-only.
    ds = _load("legacy-datasets/banking77", "train+test")
    labels = clean_label_names(ds.features["label"].names, "banking77")
    rows = []
    for i, row in enumerate(ds):
        opts, gold = sample_options(labels, labels[row["label"]], rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", row["text"], "Which banking intent does the message express?",
                             opts, False, gold, "banking77", f"banking77-{i}"))
    return rows


def ag_news(rng):
    ds = _load("fancyzhx/ag_news", "train[:12000]+test[:3000]")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return [Decision("choice", row["text"], "What is the topic of this article?",
                     list(labels), False, int(row["label"]), "ag_news", f"ag_news-{i}")
            for i, row in enumerate(ds)]


def mnli(rng):
    ds = _load("nyu-mll/multi_nli", "train[:12000]+validation_matched[:3000]")
    labels = ["entailment", "neutral", "contradiction"]
    rows = []
    for i, row in enumerate(ds):
        if row["label"] < 0:
            continue
        rows.append(Decision("choice", row["premise"],
                             f'What is the relation of the context to the hypothesis: "{row["hypothesis"]}"?',
                             list(labels), False, int(row["label"]), "mnli", f"mnli-{i}"))
    return rows


def sst5(rng):
    ds = _load("SetFit/sst5", "train+validation+test")
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    return [Decision("score", row["text"], "What is the sentiment of the text?",
                     list(levels), True, int(row["label"]), "sst5", f"sst5-{i}")
            for i, row in enumerate(ds)]


def yelp(rng):
    ds = _load("Yelp/yelp_review_full", "train[:12000]+test[:3000]")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    return [Decision("score", row["text"][:1500], "How many stars did the reviewer give?",
                     list(levels), True, int(row["label"]), "yelp", f"yelp-{i}")
            for i, row in enumerate(ds)]


def boolq(rng):
    ds = _load("google/boolq", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        proposition = row["question"].strip().rstrip("?") + "?"
        rows.append(Decision("noul", row["passage"][:1500],
                             f"Is the answer to this question yes: {proposition}",
                             ["true", "false"], False, 0 if row["answer"] else 1, "boolq", f"boolq-{i}"))
    return rows


def triage(rng):
    return make_triage(4000, rng)


# ---- commonsense & reasoning ------------------------------------------------------------------

def social_iqa(rng):
    ds = _load("allenai/social_i_qa", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(row[k]) for k in ("answerA", "answerB", "answerC")]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(f'{row["context"]} {row["question"]}'),
                             "Which answer is most appropriate?", opts, False,
                             int(row["label"]) - 1, "social_iqa", f"social_iqa-{i}"))
    return rows


def cosmos_qa(rng):
    ds = _load("allenai/cosmos_qa", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(row[k]) for k in ("answer0", "answer1", "answer2", "answer3")]
        if not usable(opts) or int(row["label"]) < 0:
            continue
        rows.append(Decision("choice", clip(oneline(row["context"])), oneline(row["question"]),
                             opts, False, int(row["label"]), "cosmos_qa", f"cosmos_qa-{i}"))
    return rows


def swag(rng):
    ds = _load("allenai/swag", "train[:25000]+validation[:5000]", name="regular")
    rows = []
    for i, row in enumerate(ds):
        if row["label"] is None or int(row["label"]) < 0:
            continue
        opts = [oneline(row[k]) for k in ("ending0", "ending1", "ending2", "ending3")]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["startphrase"]), "Which ending is most plausible?",
                             opts, False, int(row["label"]), "swag", f"swag-{i}"))
    return rows


def quartz(rng):
    ds = _load("allenai/quartz", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        texts = [oneline(t) for t in row["choices"]["text"]]
        if not row["answerKey"] or not usable(texts):
            continue
        rows.append(Decision("choice", clip(oneline(f'{row["para"]} {row["question"]}')),
                             "Which option is correct?", texts, False,
                             key_index(row["choices"]["label"], row["answerKey"]), "quartz", f"quartz-{i}"))
    return rows


def winogrande(rng):
    ds = _load("allenai/winogrande", "train+validation", name="winogrande_xl")
    rows = []
    for i, row in enumerate(ds):
        if not row["answer"]:
            continue
        opts = [oneline(row["option1"]), oneline(row["option2"])]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["sentence"]), "Which option fills the blank?",
                             opts, False, int(row["answer"]) - 1, "winogrande", f"winogrande-{i}"))
    return rows


def logiqa(rng):
    ds = _load("lucasmccabe/logiqa", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(o) for o in row["options"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", clip(oneline(f'{row["context"]} {row["query"]}')),
                             "Which option is correct?", opts, False,
                             int(row["correct_option"]), "logiqa", f"logiqa-{i}"))
    return rows


def reclor(rng):
    ds = _load("metaeval/reclor", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(a) for a in row["answers"]]
        if not usable(opts) or int(row["label"]) < 0:
            continue
        rows.append(Decision("choice", clip(oneline(f'{row["context"]} {row["question"]}')),
                             "Which option is correct?", opts, False,
                             int(row["label"]), "reclor", f"reclor-{i}"))
    return rows


def codah(rng):
    ds = _load("jaredfern/codah", "train", name="codah")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(a) for a in row["candidate_answers"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["question_propmt"]), "Which completion is most plausible?",
                             opts, False, int(row["correct_answer_idx"]), "codah", f"codah-{i}"))
    return rows


def hellaswag(rng):
    ds = _load("Rowan/hellaswag", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        if row["label"] in ("", None):
            continue
        opts = [oneline(e) for e in row["endings"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", clip(oneline(row["ctx"])), "Which ending is most plausible?",
                             opts, False, int(row["label"]), "hellaswag", f"hellaswag-{i}"))
    return rows


def piqa(rng):
    ds = _load("ybisk/piqa", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        if int(row["label"]) < 0:
            continue
        opts = [oneline(row["sol1"]), oneline(row["sol2"])]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["goal"]), "Which solution is more appropriate?",
                             opts, False, int(row["label"]), "piqa", f"piqa-{i}"))
    return rows


# ---- science & knowledge exams ----------------------------------------------------------------

def mmlu(rng):
    ds = _load("cais/mmlu", "test", name="all")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(c) for c in row["choices"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             opts, False, int(row["answer"]), "mmlu", f"mmlu-{i}"))
    return rows


def mmlu_pro(rng):
    ds = _load("TIGER-Lab/MMLU-Pro", "test")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(o) for o in row["options"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             opts, False, int(row["answer_index"]), "mmlu_pro", f"mmlu_pro-{i}"))
    return rows


def arc(rng):
    rows = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = _load("allenai/ai2_arc", "train+validation+test", name=cfg)
        for i, row in enumerate(ds):
            texts = [oneline(t) for t in row["choices"]["text"]]
            if not row["answerKey"] or not usable(texts):
                continue
            gold = key_index(row["choices"]["label"], row["answerKey"])
            rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                                 texts, False, gold, "arc", f"arc-{cfg}-{i}"))
    return rows


def openbookqa(rng):
    ds = _load("allenai/openbookqa", "train+validation+test", name="main")
    rows = []
    for i, row in enumerate(ds):
        texts = [oneline(t) for t in row["choices"]["text"]]
        if not row["answerKey"] or not usable(texts):
            continue
        rows.append(Decision("choice", oneline(row["question_stem"]), "Which option is correct?",
                             texts, False, key_index(row["choices"]["label"], row["answerKey"]),
                             "openbookqa", f"openbookqa-{i}"))
    return rows


def commonsense_qa(rng):
    ds = _load("tau/commonsense_qa", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        texts = [oneline(t) for t in row["choices"]["text"]]
        if not row["answerKey"] or not usable(texts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             texts, False, key_index(row["choices"]["label"], row["answerKey"]),
                             "commonsense_qa", f"commonsense_qa-{i}"))
    return rows


def sciq(rng):
    ds = _load("allenai/sciq", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(row["correct_answer"]), oneline(row["distractor1"]),
                oneline(row["distractor2"]), oneline(row["distractor3"])]
        if not usable(opts):
            continue
        ctx = clip(f'{row["support"]} {row["question"]}'.strip())
        rows.append(Decision("choice", ctx, row["question"], opts, False, 0, "sciq", f"sciq-{i}"))
    return rows


def qasc(rng):
    ds = _load("allenai/qasc", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        texts = [oneline(t) for t in row["choices"]["text"]]
        if not row["answerKey"] or not usable(texts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             texts, False, key_index(row["choices"]["label"], row["answerKey"]),
                             "qasc", f"qasc-{i}"))
    return rows


def medmcqa(rng):
    ds = _load("openlifescienceai/medmcqa", "validation")
    rows = []
    for i, row in enumerate(ds):
        if row.get("choice_type") == "multi":
            continue
        opts = [oneline(row[k]) for k in ("opa", "opb", "opc", "opd")]
        gold = int(row["cop"])
        if not usable(opts) or not 0 <= gold < 4:
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             opts, False, gold, "medmcqa", f"medmcqa-{i}"))
    return rows


def medqa(rng):
    ds = _load("GBaker/MedQA-USMLE-4-options", "train+test")
    rows = []
    for i, row in enumerate(ds):
        options = row["options"]
        keys = sorted(options.keys())
        opts = [oneline(options[k]) for k in keys]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             opts, False, keys.index(str(row["answer_idx"]).strip()),
                             "medqa", f"medqa-{i}"))
    return rows


def head_qa(rng):
    # The head_qa script is broken on current huggingface_hub; the auto-generated parquet export
    # exposes the same English schema under the 'default' config.
    ds = _load("dvilares/head_qa", "train+validation+test", name="default",
               revision="refs/convert/parquet")
    rows = []
    for i, row in enumerate(ds):
        answers = row["answers"]
        opts = [oneline(a["atext"]) for a in answers]
        aids = [int(a["aid"]) for a in answers]
        gold = int(row["ra"])
        if not usable(opts) or gold not in aids:
            continue
        rows.append(Decision("choice", oneline(row["qtext"]), "Which option is correct?",
                             opts, False, aids.index(gold), "head_qa", f"head_qa-{i}"))
    return rows


def gpqa(rng):
    ds = _load("Idavidrein/gpqa", "train", name="gpqa_main")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(row["Correct Answer"]), oneline(row["Incorrect Answer 1"]),
                oneline(row["Incorrect Answer 2"]), oneline(row["Incorrect Answer 3"])]
        if not usable(opts):
            continue
        rows.append(Decision("choice", clip(oneline(row["Question"])), "Which option is correct?",
                             opts, False, 0, "gpqa", f"gpqa-{i}"))
    return rows


# ---- math (multiple choice) -------------------------------------------------------------------

def aqua_rat(rng):
    ds = _load("deepmind/aqua_rat", "train+validation+test", name="raw")
    rows = []
    for i, row in enumerate(ds):
        opts = [re.sub(r"^[A-Ea-e]\s*\)\s*", "", oneline(o)) for o in row["options"]]
        gold = letter_index(row["correct"])
        if not usable(opts) or not 0 <= gold < len(opts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which option is correct?",
                             opts, False, gold, "aqua_rat", f"aqua_rat-{i}"))
    return rows


def math_qa(rng):
    ds = _load("allenai/math_qa", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        parsed = parse_packed(row["options"])
        present = [k for k in "abcde" if k in parsed]
        opts = [parsed[k] for k in present]
        if not usable(opts) or row["correct"] not in parsed:
            continue
        rows.append(Decision("choice", row["Problem"], "Which option is correct?", opts, False,
                             present.index(row["correct"]), "math_qa", f"math_qa-{i}"))
    return rows


# ---- reading comprehension --------------------------------------------------------------------

def race(rng):
    ds = _load("ehovy/race", "train[:20000]+validation[:3000]+test[:3000]", name="all")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(o) for o in row["options"]]
        if not usable(opts):
            continue
        rows.append(Decision("choice", clip(oneline(row["article"])), oneline(row["question"]),
                             opts, False, letter_index(row["answer"]), "race", f"race-{i}"))
    return rows


def quail(rng):
    ds = _load("textmachinelab/quail", "train+validation")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(a) for a in row["answers"]]
        gold = int(row["correct_answer_id"])
        if not usable(opts) or not 0 <= gold < len(opts):
            continue
        rows.append(Decision("choice", clip(oneline(row["context"])), oneline(row["question"]),
                             opts, False, gold, "quail", f"quail-{i}"))
    return rows


def dream(rng):
    ds = _load("dataset-org/dream", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(c) for c in row["choice"]]
        if not usable(opts):
            continue
        ctx = clip(oneline(" ".join(row["dialogue"])))
        rows.append(Decision("choice", ctx, row["question"], opts, False,
                             text_index(opts, row["answer"]), "dream", f"dream-{i}"))
    return rows


# ---- natural language inference ---------------------------------------------------------------

_ENTAILMENT = ["entailment", "neutral", "contradiction"]


def snli(rng):
    ds = _load("stanfordnlp/snli", "train[:30000]")
    rows = []
    for i, row in enumerate(ds):
        if int(row["label"]) < 0:
            continue
        rows.append(Decision("choice", oneline(row["premise"]),
                             f'What is the relation to the hypothesis: "{oneline(row["hypothesis"])}"?',
                             list(_ENTAILMENT), False, int(row["label"]), "snli", f"snli-{i}"))
    return rows


def anli(rng):
    ds = _load("facebook/anli", "dev_r1+dev_r2+dev_r3")
    rows = []
    for i, row in enumerate(ds):
        if int(row["label"]) < 0:
            continue
        rows.append(Decision("choice", clip(oneline(row["premise"])),
                             f'What is the relation to the hypothesis: "{oneline(row["hypothesis"])}"?',
                             list(_ENTAILMENT), False, int(row["label"]), "anli", f"anli-{i}"))
    return rows


def scitail(rng):
    ds = _load("allenai/scitail", "train+validation+test", name="tsv_format")
    rows = []
    for i, row in enumerate(ds):
        entailed = str(row["label"]).strip().lower() == "entails"
        rows.append(Decision("noul", clip(oneline(row["premise"])),
                             f'Does the premise entail: "{oneline(row["hypothesis"])}"?',
                             ["true", "false"], False, 0 if entailed else 1, "scitail", f"scitail-{i}"))
    return rows


def folio(rng):
    ds = _load("tasksource/folio", "train+validation")
    labels = ["True", "False", "Uncertain"]
    lookup = {"true": 0, "false": 1, "uncertain": 2}
    rows = []
    for i, row in enumerate(ds):
        key = str(row["label"]).strip().lower()
        if key not in lookup:
            continue
        rows.append(Decision("choice", clip(oneline(row["premises"])),
                             f'Is the conclusion true, false, or uncertain: "{oneline(row["conclusion"])}"?',
                             list(labels), False, lookup[key], "folio", f"folio-{i}"))
    return rows


def glue_qnli(rng):
    ds = _load("nyu-mll/glue", "validation", name="qnli")
    rows = []
    for i, row in enumerate(ds):
        # QNLI label 0 = entailment, 1 = not_entailment
        rows.append(Decision("noul", oneline(row["sentence"]),
                             f'Does the sentence answer this question: "{oneline(row["question"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 0 else 1,
                             "glue_qnli", f"glue_qnli-{i}"))
    return rows


def glue_rte(rng):
    ds = _load("nyu-mll/glue", "validation", name="rte")
    rows = []
    for i, row in enumerate(ds):
        # RTE label 0 = entailment, 1 = not_entailment
        rows.append(Decision("noul", oneline(row["sentence1"]),
                             f'Does it entail: "{oneline(row["sentence2"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 0 else 1,
                             "glue_rte", f"glue_rte-{i}"))
    return rows


def superglue_cb(rng):
    ds = _load("aps/super_glue", "validation", name="cb")
    labels = ["entailment", "contradiction", "neutral"]
    rows = []
    for i, row in enumerate(ds):
        if int(row["label"]) < 0:
            continue
        rows.append(Decision("choice", clip(oneline(row["premise"])),
                             f'What is the relation to the hypothesis: "{oneline(row["hypothesis"])}"?',
                             list(labels), False, int(row["label"]), "superglue_cb", f"superglue_cb-{i}"))
    return rows


def superglue_copa(rng):
    ds = _load("aps/super_glue", "validation", name="copa")
    rows = []
    for i, row in enumerate(ds):
        if int(row["label"]) < 0:
            continue
        opts = [oneline(row["choice1"]), oneline(row["choice2"])]
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(f'{row["premise"]} {row["question"]}'),
                             "Which choice is more plausible?", opts, False,
                             int(row["label"]), "superglue_copa", f"superglue_copa-{i}"))
    return rows


# ---- fact verification & truthfulness ---------------------------------------------------------

def vitaminc(rng):
    ds = _load("tals/vitaminc", "train[:25000]+validation[:5000]")
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    lookup = {n: j for j, n in enumerate(labels)}
    rows = []
    for i, row in enumerate(ds):
        key = str(row["label"]).strip().upper()
        if key not in lookup:
            continue
        rows.append(Decision("choice", clip(oneline(row["evidence"])),
                             f'What is the relation of the evidence to the claim: "{oneline(row["claim"])}"?',
                             list(labels), False, lookup[key], "vitaminc", f"vitaminc-{i}"))
    return rows


def fever(rng):
    ds = _load("fever/fever", "train[:30000]", name="v1.0")
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    lookup = {n: j for j, n in enumerate(labels)}
    rows = []
    for i, row in enumerate(ds):
        key = str(row["label"]).strip().upper()
        if key not in lookup:
            continue
        rows.append(Decision("choice", oneline(row["claim"]),
                             "Is this claim supported, refuted, or is there not enough info?",
                             list(labels), False, lookup[key], "fever", f"fever-{i}"))
    return rows


def health_fact(rng):
    ds = _load("ImperialCollegeLondon/health_fact", "train+validation+test")
    labels = ["false", "mixture", "true", "unproven"]
    rows = []
    for i, row in enumerate(ds):
        gold = int(row["label"])
        if not 0 <= gold < 4:
            continue
        ctx = clip(oneline(row["main_text"]))
        if not ctx.strip():
            continue
        rows.append(Decision("choice", ctx,
                             f'What is the veracity of the claim: "{oneline(row["claim"])}"?',
                             list(labels), False, gold, "health_fact", f"health_fact-{i}"))
    return rows


def pubmedqa(rng):
    ds = _load("qiaojin/PubMedQA", "train", name="pqa_labeled")
    labels = ["yes", "no", "maybe"]
    lookup = {n: j for j, n in enumerate(labels)}
    rows = []
    for i, row in enumerate(ds):
        decision = str(row["final_decision"]).strip().lower()
        if decision not in lookup:
            continue
        contexts = row["context"]
        if isinstance(contexts, dict):
            contexts = contexts.get("contexts", [])
        ctx = clip(oneline(" ".join(contexts)))
        rows.append(Decision("choice", ctx, oneline(row["question"]),
                             list(labels), False, lookup[decision], "pubmedqa", f"pubmedqa-{i}"))
    return rows


def tab_fact(rng):
    ds = _load("ibm-research/tab_fact", "train[:25000]+validation", name="tab_fact")
    rows = []
    for i, row in enumerate(ds):
        entailed = int(row["label"]) == 1  # 1 = entailed, 0 = refuted
        table = row["table"]  # {header: [...], rows: [[...]], caption: str}
        header = " | ".join(str(h) for h in table.get("header", []))
        body = "; ".join(" , ".join(str(c) for c in r) for r in table.get("rows", []))
        ctx = clip(oneline(f'{table.get("caption", "")} {header} {body}'))
        if not ctx.strip():
            continue
        rows.append(Decision("noul", ctx, f'Does the table support: "{oneline(row["statement"])}"?',
                             ["true", "false"], False, 0 if entailed else 1, "tab_fact", f"tab_fact-{i}"))
    return rows


def halueval(rng):
    ds = _load("pminervini/HaluEval", "data", name="qa")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(f'{row.get("knowledge", "")} {row["question"]}'.strip()))
        if not ctx.strip():
            continue
        # a hallucinated answer -> "true" (index 0); a faithful answer -> "false" (index 1)
        for tag, answer, gold in (("r", row["right_answer"], 1), ("h", row["hallucinated_answer"], 0)):
            rows.append(Decision("noul", ctx,
                                 f'Is this answer hallucinated (unsupported): "{oneline(answer)}"?',
                                 ["true", "false"], False, gold, "halueval", f"halueval-{i}-{tag}"))
    return rows


def squad_v2(rng):
    # The parquet uses a datasets>=4 'List' feature type that datasets 3.x cannot parse and the
    # legacy script imports a removed module, so read the parquet files off the hub directly.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "rajpurkar/squad_v2"
    parquet = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                     if f.endswith(".parquet") and ("train" in f or "validation" in f))
    rows = []
    n = 0
    for f in parquet:
        for row in pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist():
            answerable = len(row["answers"]["text"]) > 0
            question = f'Does the passage answer this question: {row["question"].strip()}'
            rows.append(Decision("noul", clip(row["context"]), question, ["true", "false"], False,
                                 0 if answerable else 1, "squad_v2", f"squad_v2-{n}"))
            n += 1
    return rows


def climate_fever(rng):
    ds = _load("tdiggelm/climate_fever", "test")
    labels = ["supports", "refutes", "not enough info", "disputed"]
    rows = []
    for i, row in enumerate(ds):
        gold = int(row["claim_label"])
        if not 0 <= gold < 4:
            continue
        rows.append(Decision("choice", oneline(row["claim"]),
                             "Is this claim supported, refuted, disputed, or is there not enough info?",
                             list(labels), False, gold, "climate_fever", f"climate_fever-{i}"))
    return rows


def truthfulqa(rng):
    ds = _load("truthfulqa/truthful_qa", "validation", name="multiple_choice")
    rows = []
    for i, row in enumerate(ds):
        opts = [oneline(c) for c in row["mc1_targets"]["choices"]]
        flags = list(row["mc1_targets"]["labels"])
        if len(opts) < 2 or 1 not in flags:
            continue
        truth = opts[flags.index(1)]
        if len(opts) > MAX_OPTIONS:
            opts, gold = sample_options(opts, truth, MAX_OPTIONS, rng)
        else:
            gold = flags.index(1)
        if not usable(opts):
            continue
        rows.append(Decision("choice", oneline(row["question"]), "Which answer is correct?",
                             opts, False, gold, "truthfulqa", f"truthfulqa-{i}"))
    return rows


# ---- paraphrase -------------------------------------------------------------------------------

def paws(rng):
    ds = _load("google-research-datasets/paws", "train[:20000]+validation+test", name="labeled_final")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["sentence1"]),
                             f'Does this sentence have the same meaning: "{oneline(row["sentence2"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1, "paws", f"paws-{i}"))
    return rows


def glue_mrpc(rng):
    ds = _load("nyu-mll/glue", "train+validation", name="mrpc")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["sentence1"]),
                             f'Does this sentence mean the same: "{oneline(row["sentence2"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "glue_mrpc", f"glue_mrpc-{i}"))
    return rows


def glue_qqp(rng):
    ds = _load("nyu-mll/glue", "validation", name="qqp")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["question1"]),
                             f'Is this question a duplicate: "{oneline(row["question2"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "glue_qqp", f"glue_qqp-{i}"))
    return rows


def med_q_pairs(rng):
    ds = _load("curaihealth/medical_questions_pairs", "train")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["question_1"]),
                             f'Is this question asking the same thing: "{oneline(row["question_2"])}"?',
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "med_q_pairs", f"med_q_pairs-{i}"))
    return rows


# ---- safety, ethics & bias --------------------------------------------------------------------

def ethics_commonsense(rng):
    ds = _load("hendrycks/ethics", "train+test", name="commonsense")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(row["input"]))
        if not ctx.strip():
            continue
        # label 1 = morally wrong, so "acceptable?" is true only when label == 0
        rows.append(Decision("noul", ctx, "Is this action morally acceptable?",
                             ["true", "false"], False, 0 if int(row["label"]) == 0 else 1,
                             "ethics_commonsense", f"ethics_commonsense-{i}"))
    return rows


def ethics_justice(rng):
    ds = _load("hendrycks/ethics", "train+test", name="justice")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(row["scenario"]))
        if not ctx.strip():
            continue
        rows.append(Decision("noul", ctx, "Is this judgment reasonable?",  # label 1 = reasonable
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "ethics_justice", f"ethics_justice-{i}"))
    return rows


def ethics_deontology(rng):
    ds = _load("hendrycks/ethics", "train+test", name="deontology")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(f'{row["scenario"]} {row["excuse"]}'))
        if not ctx.strip():
            continue
        rows.append(Decision("noul", ctx, "Is this excuse reasonable?",  # label 1 = reasonable
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "ethics_deontology", f"ethics_deontology-{i}"))
    return rows


def ethics_virtue(rng):
    ds = _load("hendrycks/ethics", "train+test", name="virtue")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(row["scenario"]))
        if not ctx.strip():
            continue
        rows.append(Decision("noul", ctx, "Does the person exhibit the stated trait?",  # label 1 = fits
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "ethics_virtue", f"ethics_virtue-{i}"))
    return rows


def moral_stories(rng):
    ds = _load("demelin/moral_stories", "train", name="full")
    rows = []
    for i, row in enumerate(ds):
        base = clip(oneline(f'{row["situation"]} {row["intention"]}'))
        # the moral action is acceptable ("true"); the immoral one is not ("false")
        for tag, action, gold in (("m", row["moral_action"], 0), ("i", row["immoral_action"], 1)):
            rows.append(Decision("noul", oneline(f'{base} Action: {oneline(action)}'),
                                 "Is this action morally acceptable?",
                                 ["true", "false"], False, gold, "moral_stories", f"moral_stories-{i}-{tag}"))
    return rows


def ethos_binary(rng):
    ds = _load("iamollas/ethos", "train", name="binary")
    rows = []
    for i, row in enumerate(ds):
        hate = round(float(row["label"])) == 1
        rows.append(Decision("noul", oneline(row["text"]), "Is this text hate speech?",
                             ["true", "false"], False, 0 if hate else 1,
                             "ethos_binary", f"ethos_binary-{i}"))
    return rows


def tweets_hate(rng):
    ds = _load("tweets-hate-speech-detection/tweets_hate_speech_detection", "train")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["tweet"]), "Is this tweet hateful?",
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "tweets_hate", f"tweets_hate-{i}"))
    return rows


# ---- sentiment, emotion & rating --------------------------------------------------------------

def imdb(rng):
    ds = _load("stanfordnlp/imdb", "train+test")  # leaves out the unlabeled 'unsupervised' split
    rows = []
    for i, row in enumerate(ds):
        label = int(row["label"])
        if label < 0:
            continue
        rows.append(Decision("noul", clip(oneline(row["text"])), "Is this review positive?",
                             ["true", "false"], False, 0 if label == 1 else 1, "imdb", f"imdb-{i}"))
    return rows


def rotten_tomatoes(rng):
    ds = _load("cornell-movie-review-data/rotten_tomatoes", "train+validation+test")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["text"]), "Is this review positive?",
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "rotten_tomatoes", f"rotten_tomatoes-{i}"))
    return rows


def amazon_polarity(rng):
    ds = _load("fancyzhx/amazon_polarity", "train[:20000]+test[:5000]")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(f'{row["title"]} {row["content"]}'))
        rows.append(Decision("noul", ctx, "Is this review positive?",
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "amazon_polarity", f"amazon_polarity-{i}"))
    return rows


def app_reviews(rng):
    ds = _load("sealuzh/app_reviews", "train[:20000]")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    rows = []
    for i, row in enumerate(ds):
        stars = int(row["star"])
        if 1 <= stars <= 5:
            rows.append(Decision("score", clip(row["review"]), "How many stars did the reviewer give?",
                                 list(levels), True, stars - 1, "app_reviews", f"app_reviews-{i}"))
    return rows


def emotion(rng):
    ds = _load("dair-ai/emotion", "train+validation+test", name="split")
    labels = clean_label_names(ds.features["label"].names, "emotion")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("choice", oneline(row["text"]), "What emotion does the text express?",
                             list(labels), False, int(row["label"]), "emotion", f"emotion-{i}"))
    return rows


def financial_phrasebank(rng):
    # The loading script builds a malformed hf:// URI and there is no parquet export, so read the
    # sentence@label lines straight out of the source zip.
    import zipfile

    from huggingface_hub import hf_hub_download
    labels = ["negative", "neutral", "positive"]
    lookup = {n: j for j, n in enumerate(labels)}
    archive = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip",
                              repo_type="dataset")
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.endswith("Sentences_AllAgree.txt"))
        lines = zf.read(member).decode("latin-1").splitlines()
    rows = []
    for i, line in enumerate(lines):
        sentence, _, label = line.rpartition("@")
        label = label.strip().lower()
        if not sentence.strip() or label not in lookup:
            continue
        rows.append(Decision("choice", oneline(sentence),
                             "What is the sentiment of this financial statement?",
                             list(labels), False, lookup[label], "financial_phrasebank",
                             f"financial_phrasebank-{i}"))
    return rows


# ---- topic & intent ---------------------------------------------------------------------------

def dbpedia_14(rng):
    ds = _load("fancyzhx/dbpedia_14", "train[:20000]+test[:5000]")
    labels = clean_label_names(ds.features["label"].names, "dbpedia_14")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(f'{row["title"]} {row["content"]}'))
        rows.append(Decision("choice", ctx, "What category does this describe?",
                             list(labels), False, int(row["label"]), "dbpedia_14", f"dbpedia_14-{i}"))
    return rows


def yahoo_topics(rng):
    ds = _load("community-datasets/yahoo_answers_topics", "train[:20000]+test[:5000]")
    labels = clean_label_names(ds.features["topic"].names, "yahoo_topics")
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(f'{row["question_title"]} {row["question_content"]} {row["best_answer"]}'))
        if not ctx.strip():
            continue
        rows.append(Decision("choice", ctx, "What topic is this question about?",
                             list(labels), False, int(row["topic"]), "yahoo_topics", f"yahoo_topics-{i}"))
    return rows


def newsgroups(rng):
    ds = _load("SetFit/20_newsgroups", "train+test")
    name_by_id: dict[int, str] = {}
    for row in ds:
        name_by_id[int(row["label"])] = oneline(row["label_text"])
    ordered_ids = sorted(name_by_id)
    labels = clean_label_names([name_by_id[k] for k in ordered_ids], "newsgroups")
    position = {k: pos for pos, k in enumerate(ordered_ids)}
    rows = []
    for i, row in enumerate(ds):
        ctx = clip(oneline(row["text"]))
        if not ctx.strip():
            continue
        rows.append(Decision("choice", ctx, "Which newsgroup is this post from?",
                             list(labels), False, position[int(row["label"])], "newsgroups", f"newsgroups-{i}"))
    return rows


def trec_coarse(rng):
    ds = _load("CogComp/trec", "train+test")
    labels = clean_label_names(ds.features["coarse_label"].names, "trec_coarse")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("choice", oneline(row["text"]), "What type of answer does this question seek?",
                             list(labels), False, int(row["coarse_label"]), "trec_coarse", f"trec_coarse-{i}"))
    return rows


def trec_fine(rng):
    ds = _load("CogComp/trec", "train+test")
    labels = clean_label_names(ds.features["fine_label"].names, "trec_fine")
    rows = []
    for i, row in enumerate(ds):
        opts, gold = sample_options(labels, labels[int(row["fine_label"])], rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", oneline(row["text"]), "What fine-grained type does this question seek?",
                             opts, False, gold, "trec_fine", f"trec_fine-{i}"))
    return rows


def clinc_oos(rng):
    ds = _load("clinc/clinc_oos", "train+validation+test", name="plus")
    labels = clean_label_names(ds.features["intent"].names, "clinc_oos")
    rows = []
    for i, row in enumerate(ds):
        opts, gold = sample_options(labels, labels[int(row["intent"])], rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", oneline(row["text"]), "What is the user's intent?",
                             opts, False, gold, "clinc_oos", f"clinc_oos-{i}"))
    return rows


def massive_intent(rng):
    ds = _load("AmazonScience/massive", "train+validation+test", name="en-US")
    labels = clean_label_names(ds.features["intent"].names, "massive_intent")
    rows = []
    for i, row in enumerate(ds):
        opts, gold = sample_options(labels, labels[int(row["intent"])], rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", oneline(row["utt"]), "What is the user's intent?",
                             opts, False, gold, "massive_intent", f"massive_intent-{i}"))
    return rows


def massive_scenario(rng):
    ds = _load("AmazonScience/massive", "train+validation+test", name="en-US")
    labels = clean_label_names(ds.features["scenario"].names, "massive_scenario")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("choice", oneline(row["utt"]), "What scenario does the utterance concern?",
                             list(labels), False, int(row["scenario"]), "massive_scenario", f"massive_scenario-{i}"))
    return rows


def subj(rng):
    ds = _load("SetFit/subj", "train+test")
    rows = []
    for i, row in enumerate(ds):
        subjective = "subj" in str(row["label_text"]).lower()
        rows.append(Decision("noul", oneline(row["text"]), "Is this sentence subjective?",
                             ["true", "false"], False, 0 if subjective else 1, "subj", f"subj-{i}"))
    return rows


def sms_spam(rng):
    ds = _load("ucirvine/sms_spam", "train")
    rows = []
    for i, row in enumerate(ds):
        rows.append(Decision("noul", oneline(row["sms"]), "Is this message spam?",
                             ["true", "false"], False, 0 if int(row["label"]) == 1 else 1,
                             "sms_spam", f"sms_spam-{i}"))
    return rows


def lang_id(rng):
    ds = _load("papluca/language-identification", "train[:20000]+validation+test")
    languages = clean_label_names(sorted(set(ds["labels"])), "lang_id")
    known = set(languages)
    rows = []
    for i, row in enumerate(ds):
        truth = oneline(row["labels"])
        if truth not in known:
            continue
        opts, gold = sample_options(languages, truth, rng.randint(4, MAX_OPTIONS), rng)
        rows.append(Decision("choice", clip(oneline(row["text"])), "What language is this text?",
                             opts, False, gold, "lang_id", f"lang_id-{i}"))
    return rows


# ---- config-bundle factories ------------------------------------------------------------------

def _tweet_eval_source(sub: str):
    is_binary = sub in ("hate", "irony", "offensive")
    prompts = {"hate": "Is this tweet hateful?", "irony": "Is this tweet ironic?",
               "offensive": "Is this tweet offensive?"}
    name = f"tweet_eval_{sub}"

    def convert(rng):
        ds = _load("cardiffnlp/tweet_eval", "train+validation+test", name=sub)
        rows = []
        if is_binary:
            for i, row in enumerate(ds):
                positive = int(row["label"]) == 1  # class 1 is the hateful/ironic/offensive one
                rows.append(Decision("noul", oneline(row["text"]), prompts[sub],
                                     ["true", "false"], False, 0 if positive else 1, name, f"{name}-{i}"))
        else:
            labels = clean_label_names(ds.features["label"].names, name)
            for i, row in enumerate(ds):
                rows.append(Decision("choice", oneline(row["text"]), "What does this tweet express?",
                                     list(labels), False, int(row["label"]), name, f"{name}-{i}"))
        return rows
    return convert


def _bbq_source(cfg: str):
    name = f"bbq_{cfg}"

    def convert(rng):
        ds = _load("heegyu/bbq", "test", name=cfg)
        rows = []
        for i, row in enumerate(ds):
            opts = [oneline(row[k]) for k in ("ans0", "ans1", "ans2")]
            gold = int(row["label"])
            if not usable(opts) or not 0 <= gold < 3:
                continue
            ctx = clip(oneline(f'{row["context"]} {row["question"]}'))
            rows.append(Decision("choice", ctx, "Which answer is correct?", opts, False, gold, name, f"{name}-{i}"))
        return rows
    return convert


def _babi_source(cfg: str):
    name = f"babi_nli_{cfg}"

    def convert(rng):
        ds = _load("tasksource/babi_nli", "train+validation", name=cfg)
        rows = []
        for i, row in enumerate(ds):
            entailed = int(row["label"]) == 1
            rows.append(Decision("noul", clip(oneline(row["premise"])),
                                 f'Does the premise entail: "{oneline(row["hypothesis"])}"?',
                                 ["true", "false"], False, 0 if entailed else 1, name, f"{name}-{i}"))
        return rows
    return convert


# ---- registry ---------------------------------------------------------------------------------

HOLDOUT_SOURCES: set[str] = {"mmlu", "medmcqa", "anli", "truthfulqa", "climate_fever", "gpqa"}
CAPS: dict[str, int] = {}  # per-source train-cap overrides; default is the build's per_source

SOURCE_LICENSE: dict[str, str] = {
    "bitext": "unstated", "banking77": "CC-BY-4.0", "ag_news": "unstated", "mnli": "unstated",
    "sst5": "unstated", "yelp": "unstated", "boolq": "CC-BY-SA-3.0", "triage": "synthetic",
    "social_iqa": "CC-BY-4.0", "cosmos_qa": "CC-BY-4.0", "swag": "unstated", "quartz": "CC-BY-4.0",
    "winogrande": "unstated", "logiqa": "unstated", "reclor": "research", "codah": "ODC-BY",
    "hellaswag": "MIT", "piqa": "AFL-3.0",
    "mmlu": "MIT", "mmlu_pro": "MIT", "arc": "CC-BY-SA-4.0", "openbookqa": "Apache-2.0",
    "commonsense_qa": "MIT", "sciq": "CC-BY-NC-3.0", "qasc": "CC-BY-4.0", "medmcqa": "Apache-2.0",
    "medqa": "unstated", "head_qa": "MIT", "gpqa": "gated/unstated",
    "aqua_rat": "Apache-2.0", "math_qa": "Apache-2.0",
    "race": "research", "quail": "CC-BY-NC-SA-4.0", "dream": "unstated",
    "snli": "CC-BY-SA-4.0", "anli": "CC-BY-NC-4.0", "scitail": "Apache-2.0", "folio": "CC/MIT",
    "glue_qnli": "permissive", "glue_rte": "permissive", "superglue_cb": "permissive",
    "superglue_copa": "permissive",
    "vitaminc": "CC-BY-SA-3.0", "fever": "CC-BY-SA-3.0", "health_fact": "MIT", "pubmedqa": "MIT",
    "tab_fact": "CC-BY-4.0", "halueval": "unstated", "squad_v2": "CC-BY-SA-4.0",
    "climate_fever": "unstated", "truthfulqa": "Apache-2.0",
    "paws": "free-to-use", "glue_mrpc": "permissive", "glue_qqp": "permissive",
    "med_q_pairs": "unstated",
    "ethics_commonsense": "MIT", "ethics_justice": "MIT", "ethics_deontology": "MIT",
    "ethics_virtue": "MIT", "moral_stories": "MIT", "ethos_binary": "AGPL/GPL",
    "tweets_hate": "GPL-3.0",
    "imdb": "permissive", "rotten_tomatoes": "unstated", "amazon_polarity": "Apache-2.0",
    "app_reviews": "unstated", "emotion": "unstated", "financial_phrasebank": "CC-BY-NC-SA-3.0",
    "dbpedia_14": "CC-BY-SA-3.0", "yahoo_topics": "unstated", "newsgroups": "unstated",
    "trec_coarse": "unstated", "trec_fine": "unstated", "clinc_oos": "CC-BY-3.0",
    "massive_intent": "CC-BY-4.0", "massive_scenario": "CC-BY-4.0", "subj": "unstated",
    "sms_spam": "unstated", "lang_id": "derived",
}

# Registration order is load-bearing: it fixes the concatenation order of the split files.
SOURCES = {
    "bitext": bitext, "banking77": banking77, "ag_news": ag_news, "mnli": mnli,
    "sst5": sst5, "yelp": yelp, "boolq": boolq, "triage": triage,
    "social_iqa": social_iqa, "cosmos_qa": cosmos_qa, "swag": swag, "quartz": quartz,
    "winogrande": winogrande, "logiqa": logiqa, "reclor": reclor, "codah": codah,
    "hellaswag": hellaswag, "piqa": piqa,
    "mmlu": mmlu, "mmlu_pro": mmlu_pro, "arc": arc, "openbookqa": openbookqa,
    "commonsense_qa": commonsense_qa, "sciq": sciq, "qasc": qasc, "medmcqa": medmcqa,
    "medqa": medqa, "head_qa": head_qa, "gpqa": gpqa,
    "aqua_rat": aqua_rat, "math_qa": math_qa,
    "race": race, "quail": quail, "dream": dream,
    "snli": snli, "anli": anli, "scitail": scitail, "folio": folio,
    "glue_qnli": glue_qnli, "glue_rte": glue_rte, "superglue_cb": superglue_cb,
    "superglue_copa": superglue_copa,
    "vitaminc": vitaminc, "fever": fever, "health_fact": health_fact, "pubmedqa": pubmedqa,
    "tab_fact": tab_fact, "halueval": halueval, "squad_v2": squad_v2,
    "climate_fever": climate_fever, "truthfulqa": truthfulqa,
    "paws": paws, "glue_mrpc": glue_mrpc, "glue_qqp": glue_qqp, "med_q_pairs": med_q_pairs,
    "ethics_commonsense": ethics_commonsense, "ethics_justice": ethics_justice,
    "ethics_deontology": ethics_deontology, "ethics_virtue": ethics_virtue,
    "moral_stories": moral_stories, "ethos_binary": ethos_binary, "tweets_hate": tweets_hate,
    "imdb": imdb, "rotten_tomatoes": rotten_tomatoes, "amazon_polarity": amazon_polarity,
    "app_reviews": app_reviews, "emotion": emotion, "financial_phrasebank": financial_phrasebank,
    "dbpedia_14": dbpedia_14, "yahoo_topics": yahoo_topics, "newsgroups": newsgroups,
    "trec_coarse": trec_coarse, "trec_fine": trec_fine, "clinc_oos": clinc_oos,
    "massive_intent": massive_intent, "massive_scenario": massive_scenario, "subj": subj,
    "sms_spam": sms_spam, "lang_id": lang_id,
}


def _register_bundles() -> None:
    for sub in ("sentiment", "emotion", "hate", "irony", "offensive"):
        name = f"tweet_eval_{sub}"
        SOURCES[name] = _tweet_eval_source(sub)
        SOURCE_LICENSE[name] = "permissive"
    try:
        from datasets import get_dataset_config_names
        for cfg in get_dataset_config_names("heegyu/bbq"):
            name = f"bbq_{cfg}"
            SOURCES[name] = _bbq_source(cfg)
            HOLDOUT_SOURCES.add(name)
            SOURCE_LICENSE[name] = "CC-BY-4.0"
        for cfg in get_dataset_config_names("tasksource/babi_nli"):
            name = f"babi_nli_{cfg}"
            SOURCES[name] = _babi_source(cfg)
            CAPS[name] = 2000
            SOURCE_LICENSE[name] = "BSD"
    except Exception:  # noqa: BLE001, S110 -- offline: bundles stay unregistered; the rest still builds
        pass


_register_bundles()
