# Benchmark Configuration Overview

| Stage      | Phase        | Benchmark                      | Type       | Shots  | Limit  | CoT   | Paradigm/Mode        | Tasks                                              |
| ---------- | ------------ | ------------------------------ | ---------- | ------ | ------ | ----- | -------------------- | -------------------------------------------------- |
| 1b         | pretraining  | MMLU                           | Harness    | 5      | 100    | No    | -                    | elementary_mathematics, high_school_biology, high_school_chemistry, high_school_physics, high_school_computer_science, high_school_us_history, high_school_geography, high_school_psychology, high_school_mathematics, high_school_government_and_politics, astronomy, world_religions, philosophy, logical_fallacies, moral_disputes |
| 1b         | sft          | MMLU                           | Harness    | 5      | 100    | No    | -                    | elementary_mathematics, high_school_biology, high_school_chemistry, high_school_physics, high_school_computer_science, high_school_us_history, high_school_geography, high_school_psychology, high_school_mathematics, high_school_government_and_politics, astronomy, world_religions, philosophy, logical_fallacies, moral_disputes |
| 1b         | pretraining  | MMLU-Pro                       | Harness    | 5      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | MMLU-Pro                       | Harness    | 5      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | GSM8K                          | Harness    | 16     | 100    | Yes   | -                    | (Default/All)                                      |
| 1b         | sft          | GSM8K                          | Harness    | 16     | 100    | Yes   | -                    | (Default/All)                                      |
| 1b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 1b         | sft          | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 1b         | pretraining  | ARC-Challenge                  | Harness    | 25     | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | ARC-Challenge                  | Harness    | 25     | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 1b         | sft          | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | IFEval                         | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | IFEval                         | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | HumanEval                      | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
| 1b         | sft          | HumanEval                      | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 1b         | sft          | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 1b         | pretraining  | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | SimpleQA_Verified              | Custom     | 0      | 50     | No    | Hard Filtering       | (Default/All)                                      |
| 1b         | sft          | SimpleQA_Verified              | Custom     | 0      | 50     | No    | Hard Filtering       | (Default/All)                                      |
| 1b         | pretraining  | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 1b         | sft          | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 1b         | pretraining  | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 1b         | sft          | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 1b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 1b         | pretraining  | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 1b         | sft          | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | MMLU                           | Harness    | 5      | 100    | No    | -                    | elementary_mathematics, high_school_biology, high_school_chemistry, high_school_physics, high_school_computer_science, high_school_us_history, high_school_geography, high_school_psychology, high_school_mathematics, high_school_government_and_politics, astronomy, world_religions, philosophy, logical_fallacies, moral_disputes |
| 3b         | sft          | MMLU                           | Harness    | 5      | 100    | No    | -                    | elementary_mathematics, high_school_biology, high_school_chemistry, high_school_physics, high_school_computer_science, high_school_us_history, high_school_geography, high_school_psychology, high_school_mathematics, high_school_government_and_politics, astronomy, world_religions, philosophy, logical_fallacies, moral_disputes |
| 3b         | pretraining  | MMLU-Pro                       | Harness    | 5      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | MMLU-Pro                       | Harness    | 5      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | GSM8K                          | Harness    | 16     | 100    | Yes   | -                    | (Default/All)                                      |
| 3b         | sft          | GSM8K                          | Harness    | 16     | 100    | Yes   | -                    | (Default/All)                                      |
| 3b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 3b         | sft          | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 3b         | pretraining  | ARC-Challenge                  | Harness    | 25     | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | ARC-Challenge                  | Harness    | 25     | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 3b         | sft          | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | IFEval                         | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | IFEval                         | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 3b         | sft          | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 3b         | pretraining  | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 3b         | sft          | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 3b         | pretraining  | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 3b         | sft          | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 3b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | sft          | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 3b         | pretraining  | L-Eval                         | Custom     | 0      | 30     | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 8b         | sft          | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 8b         | pretraining  | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 8b         | sft          | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 8b         | sft          | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 8b         | pretraining  | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 8b         | sft          | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 8b         | pretraining  | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 8b         | sft          | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 8b         | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | sft          | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | GPQA Diamond                   | Harness    | 0      | 50     | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | AIME 2025                      | Custom     | 0      | 75     | No    | 8-shot CoT           | (Default/All)                                      |
| 8b         | pretraining  | L-Eval                         | Custom     | 0      | 30     | No    | -                    | (Default/All)                                      |
| 8b         | pretraining  | RULER                          | Custom     | 0      | 30     | No    | -                    | longchat_qa                                        |
| 8b         | pretraining  | SWE-bench Verified             | Custom     | 0      | 30     | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 70b        | sft          | BBH (Big Bench Hard)           | Harness    | 3      | 50     | Yes   | -                    | (Default/All)                                      |
| 70b        | pretraining  | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 70b        | sft          | MATH                           | Harness    | 8      | 50     | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 70b        | sft          | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| 70b        | pretraining  | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 70b        | sft          | IndicGLUE                      | Custom     | 0      | All    | No    | -                    | indic-wnli, indic-copa, xquad-in                   |
| 70b        | pretraining  | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 70b        | sft          | IndicQA                        | Custom     | 0      | All    | No    | 5-10 shot            | (Default/All)                                      |
| 70b        | pretraining  | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | Indic-Bias (FairITales)        | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | sft          | HELM Safety                    | Custom     | 0      | All    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | GPQA Diamond                   | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | AIME 2025                      | Custom     | 0      | 100    | No    | 8-shot CoT           | (Default/All)                                      |
| 70b        | pretraining  | L-Eval                         | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
| 70b        | pretraining  | RULER                          | Custom     | 0      | 50     | No    | -                    | longchat_qa                                        |
| 70b        | pretraining  | SWE-bench Verified             | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | MMLU                           | Harness    | 0      | 250    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | MMLU-Pro                       | Harness    | 0      | 250    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | TriviaQA                       | Harness    | 0      | 250    | Yes   | -                    | (Default/All)                                      |
| sft_70b    | sft          | GSM8K                          | Harness    | 0      | 250    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | BBH (Big Bench Hard)           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | ARC-Challenge                  | Harness    | 0      | 250    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | MATH                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | IFEval                         | Harness    | 0      | 250    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | HumanEval                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | MSGS                           | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | BLiMP                          | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| sft_70b    | sft          | TruthfulQA                     | Harness    | 0      | 100    | No    | MC2                  | (Default/All)                                      |
| sft_70b    | pretraining  | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | HellaSwag                      | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | Winogrande                     | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | PIQA                           | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | pretraining  | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | LAMBADA                        | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | SimpleQA_Verified              | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | IndicGLUE                      | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | IndicQA                        | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | Indic-Bias (FairITales)        | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | HELM Safety                    | Harness    | 0      | All    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | GPQA Diamond                   | Harness    | 0      | 100    | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | AIME 2025                      | Custom     | 0      | 100    | No    | 8-shot CoT           | (Default/All)                                      |
| sft_70b    | sft          | L-Eval                         | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
| sft_70b    | sft          | RULER                          | Custom     | 0      | 50     | No    | -                    | longchat_qa                                        |
| sft_70b    | sft          | SWE-bench Verified             | Custom     | 0      | 50     | No    | -                    | (Default/All)                                      |
