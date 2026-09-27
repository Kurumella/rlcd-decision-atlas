# RLCD Decision Atlas — full source table

The per-source companion to the README. Each source is converted to single-gold-answer typed questions (`choice` / ordered `score` / `noul` true-false). Holdout (OOD) sources go **only** to `holdout.jsonl`; all others split context-grouped into train/val/test. Licenses are recorded as metadata only — no source is filtered by license.

## Totals

- **train:** 531,088
- **val:** 73,864
- **test:** 73,864
- **holdout:** 27,548
- **sources built:** 114  |  **errored/skipped:** 1

## Sources

| source | HF id (config) | primitive | K | role | license | train | val | test | holdout |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | fancyzhx/ag_news | choice | 4 | train | unstated | 8000 | 1000 | 1000 |  |
| amazon_polarity | fancyzhx/amazon_polarity | noul | 2 | train | Apache-2.0 | 8000 | 1000 | 1000 |  |
| anli | facebook/anli | choice |  | holdout | CC-BY-NC-4.0 |  |  |  | 2000 |
| app_reviews | sealuzh/app_reviews | score | 5 | train | unstated | 8000 | 1000 | 1000 |  |
| aqua_rat | deepmind/aqua_rat (raw) | choice | 5 | train | Apache-2.0 | 8000 | 1000 | 1000 |  |
| arc | allenai/ai2_arc (Challenge+Easy) | choice | 4 | train | CC-BY-SA-4.0 | 5787 | 1000 | 1000 |  |
| babi_nli_agents-motivations | tasksource/babi_nli (agents-motivations) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_basic-coreference | tasksource/babi_nli (basic-coreference) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_basic-deduction | tasksource/babi_nli (basic-deduction) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_basic-induction | tasksource/babi_nli (basic-induction) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_compound-coreference | tasksource/babi_nli (compound-coreference) | noul | 2 | train | BSD | 1199 | 150 | 150 |  |
| babi_nli_conjunction | tasksource/babi_nli (conjunction) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_counting | tasksource/babi_nli (counting) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_indefinite-knowledge | tasksource/babi_nli (indefinite-knowledge) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_lists-sets | tasksource/babi_nli (lists-sets) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_path-finding | tasksource/babi_nli (path-finding) | noul | 2 | train | BSD | 1199 | 150 | 150 |  |
| babi_nli_positional-reasoning | tasksource/babi_nli (positional-reasoning) | noul | 2 | train | BSD | 1196 | 150 | 150 |  |
| babi_nli_simple-negation | tasksource/babi_nli (simple-negation) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_single-supporting-fact | tasksource/babi_nli (single-supporting-fact) | noul | 2 | train | BSD | 1199 | 150 | 150 |  |
| babi_nli_size-reasoning | tasksource/babi_nli (size-reasoning) | noul | 2 | train | BSD | 1199 | 150 | 150 |  |
| babi_nli_three-arg-relations | tasksource/babi_nli (three-arg-relations) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_three-supporting-facts | tasksource/babi_nli (three-supporting-facts) | noul | 2 | train | BSD | 1198 | 150 | 150 |  |
| babi_nli_time-reasoning | tasksource/babi_nli (time-reasoning) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_two-arg-relations | tasksource/babi_nli (two-arg-relations) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_two-supporting-facts | tasksource/babi_nli (two-supporting-facts) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| babi_nli_yes-no-questions | tasksource/babi_nli (yes-no-questions) | noul | 2 | train | BSD | 1200 | 150 | 150 |  |
| banking77 | legacy-datasets/banking77 | choice | 5 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| bbq_Age | heegyu/bbq (Age) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Disability_status | heegyu/bbq (Disability_status) | choice |  | holdout | CC-BY-4.0 |  |  |  | 1556 |
| bbq_Gender_identity | heegyu/bbq (Gender_identity) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Nationality | heegyu/bbq (Nationality) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Physical_appearance | heegyu/bbq (Physical_appearance) | choice |  | holdout | CC-BY-4.0 |  |  |  | 1576 |
| bbq_Race_ethnicity | heegyu/bbq (Race_ethnicity) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Race_x_SES | heegyu/bbq (Race_x_SES) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Race_x_gender | heegyu/bbq (Race_x_gender) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Religion | heegyu/bbq (Religion) | choice |  | holdout | CC-BY-4.0 |  |  |  | 1200 |
| bbq_SES | heegyu/bbq (SES) | choice |  | holdout | CC-BY-4.0 |  |  |  | 2000 |
| bbq_Sexual_orientation | heegyu/bbq (Sexual_orientation) | choice |  | holdout | CC-BY-4.0 |  |  |  | 864 |
| bitext | bitext/Bitext-customer-support-llm-chatbot-training-dataset | choice | 9 | train | unstated | 8000 | 1000 | 1000 |  |
| boolq | google/boolq | noul | 2 | train | CC-BY-SA-3.0 | 8000 | 1000 | 1000 |  |
| climate_fever | tdiggelm/climate_fever | choice |  | holdout | unstated |  |  |  | 1535 |
| clinc_oos | clinc/clinc_oos (plus) | choice | 4 | train | CC-BY-3.0 | 8000 | 1000 | 1000 |  |
| codah | jaredfern/codah (codah) | choice | 4 | train | ODC-BY | 2219 | 277 | 277 |  |
| commonsense_qa | tau/commonsense_qa | choice | 5 | train | MIT | 8000 | 1000 | 1000 |  |
| cosmos_qa | allenai/cosmos_qa | choice | 4 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| dbpedia_14 | fancyzhx/dbpedia_14 | choice | 14 | train | CC-BY-SA-3.0 | 8000 | 1000 | 1000 |  |
| dream | dataset-org/dream | choice | 3 | train | unstated | 8000 | 1000 | 1000 |  |
| emotion | dair-ai/emotion (split) | choice | 6 | train | unstated | 8000 | 1000 | 1000 |  |
| ethics_commonsense | hendrycks/ethics (commonsense) | noul | 2 | train | MIT | 8000 | 1000 | 1000 |  |
| ethics_deontology | hendrycks/ethics (deontology) | noul | 2 | train | MIT | 8000 | 1000 | 1000 |  |
| ethics_justice | hendrycks/ethics (justice) | noul | 2 | train | MIT | 8000 | 1000 | 1000 |  |
| ethics_virtue | hendrycks/ethics (virtue) | noul | 2 | train | MIT | 8000 | 1000 | 1000 |  |
| ethos_binary | iamollas/ethos (binary) | noul | 2 | train | AGPL/GPL | 800 | 99 | 99 |  |
| fever | fever/fever (v1.0) | choice | 3 | train | CC-BY-SA-3.0 | 8000 | 1000 | 1000 |  |
| financial_phrasebank | takala/financial_phrasebank (sentences_allagree) | choice | 3 | train | CC-BY-NC-SA-3.0 | 1812 | 226 | 226 |  |
| folio | tasksource/folio | choice | 3 | train | CC/MIT | 964 | 120 | 120 |  |
| glue_mrpc | nyu-mll/glue (mrpc) | noul | 2 | train | permissive | 2076 | 1000 | 1000 |  |
| glue_qnli | nyu-mll/glue (qnli) | noul | 2 | train | permissive | 3462 | 1000 | 1000 |  |
| glue_qqp | nyu-mll/glue (qqp) | noul | 2 | train | permissive | 8000 | 1000 | 1000 |  |
| glue_rte | nyu-mll/glue (rte) | noul | 2 | train | permissive | 223 | 27 | 27 |  |
| halueval | pminervini/HaluEval (qa) | noul | 2 | train | unstated | 8000 | 1000 | 1000 |  |
| head_qa | dvilares/head_qa | choice | 5 | train | MIT | 8000 | 1000 | 1000 |  |
| health_fact | ImperialCollegeLondon/health_fact | choice | 4 | train | MIT | 8000 | 1000 | 1000 |  |
| hellaswag | Rowan/hellaswag | choice | 4 | train | MIT | 8000 | 1000 | 1000 |  |
| imdb | stanfordnlp/imdb | noul | 2 | train | permissive | 8000 | 1000 | 1000 |  |
| lang_id | papluca/language-identification | choice | 12 | train | derived | 8000 | 1000 | 1000 |  |
| logiqa | lucasmccabe/logiqa | choice | 4 | train | unstated | 6634 | 1000 | 1000 |  |
| massive_intent | AmazonScience/massive (en-US) | choice | 18 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| massive_scenario | AmazonScience/massive (en-US) | choice | 18 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| math_qa | allenai/math_qa | choice | 5 | train | Apache-2.0 | 8000 | 1000 | 1000 |  |
| med_q_pairs | curaihealth/medical_questions_pairs | noul | 2 | train | unstated | 1048 | 1000 | 1000 |  |
| medmcqa | openlifescienceai/medmcqa | choice |  | holdout | Apache-2.0 |  |  |  | 2000 |
| medqa | GBaker/MedQA-USMLE-4-options | choice | 4 | train | unstated | 8000 | 1000 | 1000 |  |
| mmlu | cais/mmlu (all) | choice |  | holdout | MIT |  |  |  | 2000 |
| mmlu_pro | TIGER-Lab/MMLU-Pro | choice | 10 | train | MIT | 8000 | 1000 | 1000 |  |
| mnli | nyu-mll/multi_nli | choice | 3 | train | unstated | 8000 | 1000 | 1000 |  |
| moral_stories | demelin/moral_stories (full) | noul | 2 | train | MIT | 8000 | 1000 | 1000 |  |
| newsgroups | SetFit/20_newsgroups | choice | 20 | train | unstated | 8000 | 1000 | 1000 |  |
| openbookqa | allenai/openbookqa (main) | choice | 4 | train | Apache-2.0 | 3954 | 1000 | 1000 |  |
| paws | google-research-datasets/paws (labeled_final) | noul | 2 | train | free-to-use | 8000 | 1000 | 1000 |  |
| piqa | ybisk/piqa | choice | 2 | train | AFL-3.0 | 8000 | 1000 | 1000 |  |
| pubmedqa | qiaojin/PubMedQA (pqa_labeled) | choice | 3 | train | MIT | 800 | 100 | 100 |  |
| qasc | allenai/qasc | choice | 8 | train | CC-BY-4.0 | 7060 | 1000 | 1000 |  |
| quail | textmachinelab/quail | choice | 4 | train | CC-BY-NC-SA-4.0 | 8000 | 1000 | 1000 |  |
| quartz | allenai/quartz | choice | 2 | train | CC-BY-4.0 | 1864 | 1000 | 1000 |  |
| race | ehovy/race (all) | choice | 4 | train | research | 8000 | 1000 | 1000 |  |
| reclor | metaeval/reclor | choice | 4 | train | research | 3138 | 1000 | 1000 |  |
| rotten_tomatoes | cornell-movie-review-data/rotten_tomatoes | noul | 2 | train | unstated | 8000 | 1000 | 1000 |  |
| sciq | allenai/sciq | choice | 4 | train | CC-BY-NC-3.0 | 8000 | 1000 | 1000 |  |
| scitail | allenai/scitail (tsv_format) | noul | 2 | train | Apache-2.0 | 8000 | 1000 | 1000 |  |
| sms_spam | ucirvine/sms_spam | noul | 2 | train | unstated | 3574 | 1000 | 1000 |  |
| snli | stanfordnlp/snli | choice | 3 | train | CC-BY-SA-4.0 | 8000 | 1000 | 1000 |  |
| social_iqa | allenai/social_i_qa | choice | 3 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| squad_v2 | rajpurkar/squad_v2 | noul | 2 | train | CC-BY-SA-4.0 | 8000 | 1000 | 1000 |  |
| sst5 | SetFit/sst5 | score | 5 | train | unstated | 8000 | 1000 | 1000 |  |
| subj | SetFit/subj | noul | 2 | train | unstated | 8000 | 1000 | 1000 |  |
| superglue_cb | aps/super_glue (cb) | choice | 3 | train | permissive | 46 | 5 | 5 |  |
| superglue_copa | aps/super_glue (copa) | choice | 2 | train | permissive | 80 | 10 | 10 |  |
| swag | allenai/swag (regular) | choice | 4 | train | unstated | 8000 | 1000 | 1000 |  |
| tab_fact | ibm-research/tab_fact (tab_fact) | noul | 2 | train | CC-BY-4.0 | 8000 | 1000 | 1000 |  |
| trec_coarse | CogComp/trec (coarse_label) | choice | 6 | train | unstated | 3952 | 1000 | 1000 |  |
| trec_fine | CogComp/trec (fine_label) | choice | 5 | train | unstated | 3952 | 1000 | 1000 |  |
| triage | (synthetic) | choice | 4 | train | synthetic | 8000 | 1000 | 1000 |  |
| truthfulqa | truthfulqa/truthful_qa (multiple_choice) | choice |  | holdout | Apache-2.0 |  |  |  | 817 |
| tweet_eval_emotion | cardiffnlp/tweet_eval (emotion) | choice | 4 | train | permissive | 3052 | 1000 | 1000 |  |
| tweet_eval_hate | cardiffnlp/tweet_eval (hate) | noul | 2 | train | permissive | 8000 | 1000 | 1000 |  |
| tweet_eval_irony | cardiffnlp/tweet_eval (irony) | noul | 2 | train | permissive | 2601 | 1000 | 1000 |  |
| tweet_eval_offensive | cardiffnlp/tweet_eval (offensive) | noul | 2 | train | permissive | 8000 | 1000 | 1000 |  |
| tweet_eval_sentiment | cardiffnlp/tweet_eval (sentiment) | choice | 3 | train | permissive | 8000 | 1000 | 1000 |  |
| tweets_hate | tweets-hate-speech-detection/tweets_hate_speech_detection | noul | 2 | train | GPL-3.0 | 8000 | 1000 | 1000 |  |
| vitaminc | tals/vitaminc | choice | 3 | train | CC-BY-SA-3.0 | 8000 | 1000 | 1000 |  |
| winogrande | allenai/winogrande (winogrande_xl) | choice | 2 | train | unstated | 8000 | 1000 | 1000 |  |
| yahoo_topics | community-datasets/yahoo_answers_topics | choice | 10 | train | unstated | 8000 | 1000 | 1000 |  |
| yelp | Yelp/yelp_review_full | score | 5 | train | unstated | 8000 | 1000 | 1000 |  |

## Errored / skipped sources

| source | license | reason |
|---|---|---|
| gpqa | gated/unstated | DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Idavidrein/gpqa to ask  |
