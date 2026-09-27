"""Regenerate DATA_CARD.md from a build's stats.json.

Usage: python scripts/make_data_card.py [data/stats.json] [DATA_CARD.md]

The card lists every source with its Hugging Face id, primitive, K, role, license, and row
counts, plus any sources that errored (e.g. gated gpqa without a token).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# logical source name -> Hugging Face id (with config in parentheses where relevant)
HFID = {
    "bitext": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    "banking77": "legacy-datasets/banking77", "ag_news": "fancyzhx/ag_news",
    "mnli": "nyu-mll/multi_nli", "sst5": "SetFit/sst5", "yelp": "Yelp/yelp_review_full",
    "boolq": "google/boolq", "triage": "(synthetic)",
    "social_iqa": "allenai/social_i_qa", "cosmos_qa": "allenai/cosmos_qa",
    "swag": "allenai/swag (regular)", "quartz": "allenai/quartz",
    "winogrande": "allenai/winogrande (winogrande_xl)", "logiqa": "lucasmccabe/logiqa",
    "reclor": "metaeval/reclor", "codah": "jaredfern/codah (codah)", "hellaswag": "Rowan/hellaswag",
    "piqa": "ybisk/piqa", "mmlu": "cais/mmlu (all)", "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "arc": "allenai/ai2_arc (Challenge+Easy)", "openbookqa": "allenai/openbookqa (main)",
    "commonsense_qa": "tau/commonsense_qa", "sciq": "allenai/sciq", "qasc": "allenai/qasc",
    "medmcqa": "openlifescienceai/medmcqa", "medqa": "GBaker/MedQA-USMLE-4-options",
    "head_qa": "dvilares/head_qa", "gpqa": "Idavidrein/gpqa (gpqa_main)",
    "aqua_rat": "deepmind/aqua_rat (raw)", "math_qa": "allenai/math_qa",
    "race": "ehovy/race (all)", "quail": "textmachinelab/quail", "dream": "dataset-org/dream",
    "snli": "stanfordnlp/snli", "anli": "facebook/anli", "scitail": "allenai/scitail (tsv_format)",
    "folio": "tasksource/folio", "glue_qnli": "nyu-mll/glue (qnli)", "glue_rte": "nyu-mll/glue (rte)",
    "superglue_cb": "aps/super_glue (cb)", "superglue_copa": "aps/super_glue (copa)",
    "vitaminc": "tals/vitaminc", "fever": "fever/fever (v1.0)",
    "health_fact": "ImperialCollegeLondon/health_fact", "pubmedqa": "qiaojin/PubMedQA (pqa_labeled)",
    "tab_fact": "ibm-research/tab_fact (tab_fact)", "halueval": "pminervini/HaluEval (qa)",
    "squad_v2": "rajpurkar/squad_v2", "climate_fever": "tdiggelm/climate_fever",
    "truthfulqa": "truthfulqa/truthful_qa (multiple_choice)",
    "paws": "google-research-datasets/paws (labeled_final)", "glue_mrpc": "nyu-mll/glue (mrpc)",
    "glue_qqp": "nyu-mll/glue (qqp)", "med_q_pairs": "curaihealth/medical_questions_pairs",
    "ethics_commonsense": "hendrycks/ethics (commonsense)", "ethics_justice": "hendrycks/ethics (justice)",
    "ethics_deontology": "hendrycks/ethics (deontology)", "ethics_virtue": "hendrycks/ethics (virtue)",
    "moral_stories": "demelin/moral_stories (full)", "ethos_binary": "iamollas/ethos (binary)",
    "tweets_hate": "tweets-hate-speech-detection/tweets_hate_speech_detection",
    "imdb": "stanfordnlp/imdb", "rotten_tomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "amazon_polarity": "fancyzhx/amazon_polarity", "app_reviews": "sealuzh/app_reviews",
    "emotion": "dair-ai/emotion (split)",
    "financial_phrasebank": "takala/financial_phrasebank (sentences_allagree)",
    "dbpedia_14": "fancyzhx/dbpedia_14", "yahoo_topics": "community-datasets/yahoo_answers_topics",
    "newsgroups": "SetFit/20_newsgroups", "trec_coarse": "CogComp/trec (coarse_label)",
    "trec_fine": "CogComp/trec (fine_label)", "clinc_oos": "clinc/clinc_oos (plus)",
    "massive_intent": "AmazonScience/massive (en-US)", "massive_scenario": "AmazonScience/massive (en-US)",
    "subj": "SetFit/subj", "sms_spam": "ucirvine/sms_spam",
    "lang_id": "papluca/language-identification",
}


def hf_id(name: str) -> str:
    if name in HFID:
        return HFID[name]
    if name.startswith("tweet_eval_"):
        return f"cardiffnlp/tweet_eval ({name[len('tweet_eval_'):]})"
    if name.startswith("bbq_"):
        return f"heegyu/bbq ({name[len('bbq_'):]})"
    if name.startswith("babi_nli_"):
        return f"tasksource/babi_nli ({name[len('babi_nli_'):]})"
    return "?"


def main() -> None:
    stats_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/stats.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "DATA_CARD.md")
    stats = json.loads(stats_path.read_text())
    total = stats.get("total", {})

    ok, errored = [], []
    for name in sorted(k for k in stats if k not in ("total", "skipped")):
        s = stats[name]
        if "error" in s:
            errored.append((name, s.get("license", "unstated"), s["error"]))
        else:
            ok.append((name, s))

    lines = [
        "# RLCD Decision Atlas — full source table",
        "",
        ("The per-source companion to the README. Each source is converted to single-gold-answer "
         "typed questions (`choice` / ordered `score` / `noul` true-false). Holdout (OOD) sources "
         "go **only** to `holdout.jsonl`; all others split context-grouped into train/val/test. "
         "Licenses are recorded as metadata only — no source is filtered by license."),
        "",
        "## Totals",
        "",
        f"- **train:** {total.get('train', 0):,}",
        f"- **val:** {total.get('val', 0):,}",
        f"- **test:** {total.get('test', 0):,}",
        f"- **holdout:** {total.get('holdout', 0):,}",
        f"- **sources built:** {len(ok)}  |  **errored/skipped:** {len(errored)}",
        "",
        "## Sources",
        "",
        "| source | HF id (config) | primitive | K | role | license | train | val | test | holdout |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in ok:
        role = s.get("role", "")
        k = s.get("k")
        lines.append(
            f"| {name} | {hf_id(name)} | {s.get('primitive') or ''} | {k if k is not None else ''} "
            f"| {role} | {s.get('license', '')} | {s.get('train', '')} | {s.get('val', '')} "
            f"| {s.get('test', '')} | {s.get('holdout', '')} |"
        )

    if errored:
        lines += ["", "## Errored / skipped sources", "",
                  "| source | license | reason |", "|---|---|---|"]
        for name, lic, err in errored:
            lines.append(f"| {name} | {lic} | {err.replace('|', '/')[:160]} |")

    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(ok)} sources, {len(errored)} errored)")


if __name__ == "__main__":
    main()
