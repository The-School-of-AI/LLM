# Benchmark Configuration Overview

| Stage      | Phase        | Benchmark                      | Type       | Shots  | CoT   | Paradigm/Mode        | Tasks                                              | 
| ---------- | ------------ | ------------------------------ | ---------- | ------ | ----- | -------------------- | -------------------------------------------------- | 
| 1b         | pretraining  | MMLU                           | Harness    | 0      | No    | -                    | elementary_mathematics, high_school_biology, high_school_chemistry, high_school_physics, high_school_computer_science, high_school_us_history, high_school_geography, high_school_psychology | 
| 1b         | pretraining  | TriviaQA                       | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | GSM8K                          | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 0      | No    | -                    | boolean_expressions, sports_understanding, word_sorting, date_understanding, object_counting | 
| 1b         | pretraining  | ARC-Challenge                  | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | MATH                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | IFEval                         | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | HumanEval                      | Custom     | 5      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | BLiMP                          | Harness    | 0      | No    | -                    | regular_plural_subject_verb_agreement_1, determiner_noun_agreement_with_adj_2, distractor_agreement_relative_clause | 
| 1b         | pretraining  | TruthfulQA                     | Harness    | 0      | No    | MC1                  | (Default/All)                                      | 
| 1b         | pretraining  | HellaSwag                      | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | Winogrande                     | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | PIQA                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | LAMBADA                        | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | SimpleQA_Verified              | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | AIME 2025                      | Custom     | 0      | No    | 0-shot (Baseline)    | (Default/All)                                      | 
| 1b         | pretraining  | IndicGLUE                      | Custom     | 0      | No    | -                    | iitp-movie-reviews, bbc-news-articles, article-genre-classification | 
| 1b         | pretraining  | IndicQA                        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | L-Eval                         | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 1b         | pretraining  | RULER                          | Custom     | 0      | No    | -                    | niah_single_1                                      | 
| 1b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | MMLU                           | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | TriviaQA                       | Harness    | 3      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | GSM8K                          | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 2      | No    | -                    | causal_judgement, disambiguation_qa, movie_recommendation, navigate, ruin_names, hyperbaton, reasoning_about_colored_objects | 
| 3b         | pretraining  | ARC-Challenge                  | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | MATH                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | IFEval                         | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | HumanEval                      | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | BLiMP                          | Harness    | 0      | No    | -                    | regular_plural_subject_verb_agreement_1, determiner_noun_agreement_with_adj_2, distractor_agreement_relative_clause | 
| 3b         | pretraining  | TruthfulQA                     | Harness    | 0      | No    | MC1                  | (Default/All)                                      | 
| 3b         | pretraining  | HellaSwag                      | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | Winogrande                     | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | PIQA                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | LAMBADA                        | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | SimpleQA_Verified              | Custom     | 0      | No    | Strict Zero-Shot     | (Default/All)                                      | 
| 3b         | pretraining  | AIME 2025                      | Custom     | 0      | No    | 3-shot               | (Default/All)                                      | 
| 3b         | pretraining  | IndicGLUE                      | Custom     | 0      | No    | -                    | iitp-movie-reviews, bbc-news-articles, article-genre-classification | 
| 3b         | pretraining  | IndicQA                        | Custom     | 0      | No    | 1-shot               | (Default/All)                                      | 
| 3b         | pretraining  | L-Eval                         | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 3b         | pretraining  | RULER                          | Custom     | 0      | No    | -                    | variable_tracking                                  | 
| 3b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | MMLU                           | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | TriviaQA                       | Harness    | 5      | Yes   | -                    | (Default/All)                                      | 
| 8b         | pretraining  | GSM8K                          | Harness    | 8      | Yes   | -                    | (Default/All)                                      | 
| 8b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 3      | Yes   | -                    | logical_deduction, tracking_shuffled_objects, geometric_shapes, formal_fallacies, multi_step_arithmetic, temporal_sequences, penguins_in_a_table | 
| 8b         | pretraining  | ARC-Challenge                  | Harness    | 25     | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | MATH                           | Harness    | 4      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | IFEval                         | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | HumanEval                      | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | MSGS                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | BLiMP                          | Harness    | 0      | No    | -                    | anaphor_gender_agreement, irregular_past_participle_adjectives, ellipsis_n_bar_2, principle_A_c_command | 
| 8b         | pretraining  | TruthfulQA                     | Harness    | 0      | No    | Generative           | (Default/All)                                      | 
| 8b         | pretraining  | HellaSwag                      | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | Winogrande                     | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | PIQA                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | LAMBADA                        | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | SimpleQA_Verified              | Custom     | 0      | No    | Uncertainty-Aware    | (Default/All)                                      | 
| 8b         | pretraining  | AIME 2025                      | Custom     | 0      | No    | 5-8 shot             | (Default/All)                                      | 
| 8b         | pretraining  | IndicGLUE                      | Custom     | 0      | No    | -                    | amnesty-ner, cvit-pib-sentence-retrieval, indic-headline-prediction | 
| 8b         | pretraining  | IndicQA                        | Custom     | 0      | No    | 3-5 shot             | (Default/All)                                      | 
| 8b         | pretraining  | L-Eval                         | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 8b         | pretraining  | RULER                          | Custom     | 0      | No    | -                    | common_words_extraction                            | 
| 8b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | MMLU                           | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | TriviaQA                       | Harness    | 5      | Yes   | -                    | (Default/All)                                      | 
| 70b        | pretraining  | GSM8K                          | Harness    | 16     | Yes   | -                    | (Default/All)                                      | 
| 70b        | pretraining  | BBH (Big Bench Hard)           | Harness    | 5      | Yes   | -                    | all_23_tasks                                       | 
| 70b        | pretraining  | ARC-Challenge                  | Harness    | 25     | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | MATH                           | Harness    | 8      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | IFEval                         | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | HumanEval                      | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | MSGS                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | BLiMP                          | Harness    | 0      | No    | -                    | wh_island, adjunct_island, complex_np_island, npi_present_1, npi_present_2, superlative_quantifiers_1 | 
| 70b        | pretraining  | TruthfulQA                     | Harness    | 0      | No    | MC2                  | (Default/All)                                      | 
| 70b        | pretraining  | HellaSwag                      | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | Winogrande                     | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | PIQA                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | LAMBADA                        | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | SimpleQA_Verified              | Custom     | 0      | No    | Hard Filtering       | (Default/All)                                      | 
| 70b        | pretraining  | AIME 2025                      | Custom     | 0      | No    | 8-shot CoT           | (Default/All)                                      | 
| 70b        | pretraining  | IndicGLUE                      | Custom     | 0      | No    | -                    | indic-wnli, indic-copa, xquad-in                   | 
| 70b        | pretraining  | IndicQA                        | Custom     | 0      | No    | 5-10 shot            | (Default/All)                                      | 
| 70b        | pretraining  | L-Eval                         | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | RULER                          | Custom     | 0      | No    | -                    | longchat_qa                                        | 
| 70b        | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | HELM Safety                    | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| 70b        | pretraining  | SWE-bench Verified             | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | MMLU                           | Harness    | 5      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | TriviaQA                       | Harness    | 5      | Yes   | -                    | (Default/All)                                      | 
| sft_70b    | sft          | GSM8K                          | Harness    | 16     | Yes   | -                    | (Default/All)                                      | 
| sft_70b    | sft          | BBH (Big Bench Hard)           | Harness    | 5      | Yes   | -                    | all_23_tasks                                       | 
| sft_70b    | sft          | ARC-Challenge                  | Harness    | 25     | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | MATH                           | Harness    | 8      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | IFEval                         | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | HumanEval                      | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | MSGS                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | BLiMP                          | Harness    | 0      | No    | -                    | wh_island, adjunct_island, complex_np_island, npi_present_1, npi_present_2, superlative_quantifiers_1 | 
| sft_70b    | sft          | TruthfulQA                     | Harness    | 0      | No    | MC2                  | (Default/All)                                      | 
| sft_70b    | sft          | HellaSwag                      | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | Winogrande                     | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | PIQA                           | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | LAMBADA                        | Harness    | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | SimpleQA_Verified              | Custom     | 0      | No    | Hard Filtering       | (Default/All)                                      | 
| sft_70b    | sft          | AIME 2025                      | Custom     | 0      | No    | 8-shot CoT           | (Default/All)                                      | 
| sft_70b    | sft          | IndicGLUE                      | Custom     | 0      | No    | -                    | indic-wnli, indic-copa, xquad-in                   | 
| sft_70b    | sft          | IndicQA                        | Custom     | 0      | No    | 5-10 shot            | (Default/All)                                      | 
| sft_70b    | sft          | L-Eval                         | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | RULER                          | Custom     | 0      | No    | -                    | longchat_qa                                        | 
| sft_70b    | sft          | Indic-Bias (FairITales)        | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | HELM Safety                    | Custom     | 0      | No    | -                    | (Default/All)                                      | 
| sft_70b    | sft          | SWE-bench Verified             | Custom     | 0      | No    | -                    | (Default/All)                                      | 
