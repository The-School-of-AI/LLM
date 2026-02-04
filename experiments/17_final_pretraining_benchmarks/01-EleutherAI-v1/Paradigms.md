### MMLU-Pro \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| MMLU-Pro | 1B | Quick Validation | Low | Basic Comprehension Failure Performance at or below random (10%) |
|  | 3B | Intermediate Assessment-1 | Medium | Reasoning Gap Detection No improvement over 1B  CoT performing worse than direct (reasoning broken) |
|  | 8B | Intermediate Assessment-2 | High | CoT performing worse than direct (reasoning broken) High variance across categories (\>30% range) |
|  | 70B | Full Production Evaluation | Very High | SFT Readiness & Capability Gaps Prompt sensitivity \>5% |

**Tip:** 

### 

### TruthfulQA \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size | Recommended Mode | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| **TruthfulQA** | **1B**  | MC1 (Single Choice) | Low | Random Hallucination |
|  | **3B** | MC1 (Single Choice) | Low | Random Hallucination |
|  | **8B** | Generative | Medium | Common Misconceptions |
|  | **70B** | MC2 (Multi-True) | High | Sophisticated Falsehoods |
|  | Referenced Paper: [https://arxiv.org/pdf/2503.19786](https://arxiv.org/pdf/2503.19786)  |  |  |  |

### BLiMP \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size | Recommended Paradigms | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| **BLiMP** | **1B \- 3B** | adj\_subject\_verb\_agreement, determiner\_noun\_agreement\_with\_adj\_2, distractor\_agreement\_relative\_clause | **Low** (Local/Surface) | Tokenizer/Early Learning: Failure here suggests poor subword mapping in Tokenizer. |
|  | **8B** | anaphor\_gender\_agreement, irregular\_past\_participle\_adjectives, ellipsis\_n\_bar\_2, principle\_A\_c\_command | **Medium** (Structural) | Structural Amnesia: The model understands words but fails to track "who is who" (state-tracking) across long-distance dependencies. |
|  | **70B** | wh\_island, adjunct\_island, complex\_np\_island, npi\_present\_1, npi\_present\_2, superlative\_quantifiers\_1 | **High** (Abstract/Logical) | Reasoning Ceiling: Success indicates a shift from "statistical guessing" to respecting hierarchical linguistic constraints (Universal Grammar). |
| **Note: All stages- Skill starts to emerge after 10M to 100M tokens of training** |  |  |  |  |

### IndicGLUE \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size | Recommended Paradigms | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| **IndicGLUE** | **1B \- 3B** | iitp-movie-reviews, bbc-news-articles, article-genre-classification | **Low** (Classification) | **Script/Tokenizer mismatch:** Failure at this size usually stems from poor script Tokenizer for Indic (e.g., Brahmic scripts). |
|  | **8B** | amnesty-ner, cvit-pib-sentence-retrieval, indic-headline-prediction | **Medium** (Retrieval/NER) | **Morphological Over-smoothing:** The model may struggle with highly inflected Indian languages, failing to link root words to their case-marked variations. |
|  | **70B** | indic-wnli (Winograd NLI), indic-copa (Choice of Plausible Alternatives), xquad-in | **High** (Reasoning/NLI) | **Transliteration/Reasoning Gap:** Even at 70B, models often rely on English-to-Indic translation pathways; failure here shows the model lacks "native" logical reasoning in Indic languages. |
| **Note: if Tokenizer is not well trained on Indic then it impacts the context length as each word will result in using high token usage.** |  |  |  |  |

### RULER \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size | Recommended Paradigms | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| **RULER** | 1B | Retrieval (Single/Multi-Key NIAH) niah\_single\_1 | Low | Recall Precision: Testing if the model can ignore noise to find a single, direct fact. |
|  | 3B | Multi-hop Tracing variable\_tracking | Medium | Contextual Drift: Assessing the model's ability to track coreference chains (e.g., A depends on B) across long text spans. |
|  | 8B | Aggregation common\_words\_extraction | High | Information Density: Evaluating if the model can summarize or count specific occurrences across the entire context window. |
|  | 70B | Question Answering (QA) longchat\_qa | Very High | Reasoning Breakdown: Testing deep synthesis where "golden passages" are buried deep within complex, distracting information. |
| **Note: if Tokenizer is not well trained on Indic then it impacts the context length as each word will result in using high token usage.** |  |  |  |  |

### 

### SimpleQA\_Verified  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size  | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| SimpleQA\_Verified  | 1B | Direct QA | Basic (Low) | Knowledge Gap: Total inability to recall niche facts. |
|  | 3B | Strict Zero-Shot | Moderate | Hallucination: Confident but incorrect factual claims. |
|  | 8B | Uncertainty-Aware | Challenging | Poor Calibration: Answering when it should abstain. |
|  | 70B | Hard Filtering | Frontier-Level | Benchmark Overfitting: Memorizing specific common artifacts. |

### HumanEval  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| HumanEval | 1B | Few-Shot / Greedy | Low (Syntactic) | Syntax Errors: Invalid Python structure or indentation. |
|  | 3B | Zero-Shot | Moderate | Logical Fallacy: Writing code that runs but calculates wrong values. |
|  | 8B | Zero-Shot / Pass@k | Competent | Edge Case Failure: Missing null checks or empty string handling. |
|  | 70B | HumanEval+ (Rigorous) | High (Algorithmic) | Data Contamination: Recalling the solution from memory instead of "reasoning." |

### AIME 2025 (NO training using this)

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| AIME 2025 | 1B | **EVALUATION ONLY (Baseline)** • 0-shot evaluation on all 30 problems • Measure: Format compliance rate • DO NOT use for training • Focus: "Can model output integers 0-999?" | **Complexity: Impossible** • Expected accuracy: 0-5% • Reasoning: Far beyond capability • Success metric: Proper format in \>50% of responses • Typical result: Random guessing with format errors | **Format Compliance Failure** \- Model cannot maintain "ANSWER: XXX" format \- Outputs non-integers or out-of-range numbers \- Generates lengthy text without clear answer \- Hallucinates steps but fails to conclude **Test Protocol:** Check if \>50% of outputs have valid integer 0-999 Manual inspection of 10 random outputs for format |
|  | 3B | **EVALUATION ONLY (Early Capability Check)** • 3-shot evaluation on all 30 problems • Measure: Basic reasoning \+ format • Compare to 1B for capability delta | **Complexity: Extremely Hard** • Expected accuracy: 5-15% • Can solve 1-4 problems (simplest ones) • Format compliance improves to \~70% • Limited multi-step reasoning (max 2-3 steps) | **Reasoning Breakdown** \- Starts correct but loses thread mid-solution \- Makes arithmetic errors in step 2-3 \- Cannot maintain logical consistency \- Repeats or loops in reasoning **Test Protocol:** Manually check 5 failures \- is logic coherent? Check arithmetic: are errors computational vs logical? |
|  | 8B | **EVALUATION ONLY (Capability Validation)** • 5-8 shot evaluation, 4 repetitions • Measure: Multi-step reasoning capability • Compare to AIME 2024 (contamination check) | **Complexity: Very Hard** • Expected accuracy: 15-30% • Can solve 4-9 problems (easier \+ some medium) • Approaching human median (27-40%) • Handles 3-5 step problems reasonably | **Contamination Risk** \- AIME 2024 score \> 2025 score by 20%+ \- Suspiciously high performance on old problems \- Perfect format on historical but not new problems **Test Protocol:** if score\_2024 \- score\_2025 \> 20: flag contamination |
|  | 70B | **EVALUATION ONLY (SOTA Benchmark)** • 8-shot CoT with self-consistency (N=40) • Tool-augmented evaluation (Python REPL) • Verification-based approach • Report: base / SC / tools separately | **Complexity: Hard (Competitive)** • Expected accuracy: 40-70% (no tools) • With self-consistency: \+10-15% • With Python tools: \+20-30% • Matches strong human competitors | **Test Set Leakage (Critical)** \- Base accuracy \>70% suggests training on AIME \- No variance across multiple runs \- Errors only on newest/hardest problems \- Perfect answers on historically difficult problems **Test Protocol:** 1\. Run on AIME 2024, 2025, and AIME Mock exams 2\. Scores should be similar (±10%) 3\. Check perplexity on problems 4\. If any red flags → assume contamination |

### MMLU  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| MMLU | 1B | Knowledge Acquisition Validation 10-15 core subjects • Elementary Math • High School Biology • High School Chemistry • High School Physics • Basic Computer Science • US History • Geography • High School Psychology | Elementary \- High School | **Catastrophic Forgetting**  • Model forgets basic knowledge during MoE architecture changes • Expert routing not learning properly • Tokenizer inefficiency on academic content **Detection:** Track accuracy on fixed 10-subject subset every 5-10B tokens. If accuracy drops \>5%, investigate immediately. **Mitigation:** Replay buffer with academic content, regular validation checkpoints |
|  | 3B | Knowledge Breadth Expansion  | High School \- Undergraduate | **Knowledge Transfer Failure**  • Growth from 1B→3B doesn't preserve learned knowledge • Poor expert specialization (all experts learn same thing) • Additional parameters not being utilized effectively **Detection:** Compare 3B vs 1B on overlapping subjects. 3B should be ≥1B \+ show improvement. **Mitigation:** Gradual scaling with intermediate checkpoints, expert diversity regularization, load balancing loss |
|  | 8B | Professional Knowledge Integration | Undergraduate \- Graduate | **Reasoning Depth & Expert Domain Gaps**  • Model memorizes facts but can't reason • Professional subjects (Law, Medicine) significantly lag • No performance improvement over 3B despite more parameters **Detection:** Test on subjects requiring multi-step inference. Analyze error patterns: knowledge gaps vs reasoning failures. **Mitigation:** Increase high-quality reasoning data, add subjects requiring synthesis, track expert utilization per domain |
|  | 70B | Comprehensive Mastery & SOTA Competitiveness | Graduate \- Expert Professional | **Benchmark Saturation & Overfitting**  • Model hits MMLU ceiling (\~90-95% max due to annotation errors) • High MMLU but poor real-world performance (gaming the benchmark) • Performance drop after SFT/RL (alignment tax) • Contamination from training data **Detection:** Full 57-subject evaluation \+ MMLU-Pro. Cross-validate with GSM8K, AIME, HumanEval. Compare pre/post-SFT scores. **Mitigation:** Avoid MMLU in training data, use MMLU-Pro/Redux for discrimination, balance alignment to prevent capability loss |

**Tip:** For intermediate checkpoints, you can use stratified sampling to maintain subject distribution while reducing evaluation time.

### TriviaQA  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| TriviaQA | 1B | **Open-Domain QA** (rc.nocontext) Zero-shot or Few-shot (0-5 shot) | **Easy-Medium Subset** • 10-25% of train (\~14K-35K) • Filter for shorter questions (\<12 tokens) • Single-entity questions preferred | **Memorization vs. Reasoning** • Test if model can distinguish "I don't know" vs. guessing • Check for verbatim training data regurgitation • Evaluate basic entity recognition • Expect EM: 15-30% • Use as baseline capability check |
|  | 3B | **Open-Domain QA** (rc.nocontext) Few-shot (3-5 shot) | **Medium Subset** • 25-50% of train (\~35K-69K) • Include multi-entity questions • Balanced difficulty distribution | **Knowledge Boundary Detection** • Overconfident incorrect answers • Retrieval hallucination • Test calibration (confidence vs. accuracy) • Evaluate on question types requiring reasoning • Check for consistent answer formats • Expect EM: 30-45% • Monitor answer normalization issues |
|  | 8B | **Open-Domain QA** (rc.nocontext) Chain-of-Thought prompting Few-shot (5-shot) | **Medium-Hard Subset** • 50-75% of train (\~69K-104K) • Include compositional questions • Multi-hop reasoning examples | **Reasoning Shortcuts** • Pattern matching vs. true reasoning • Context length handling • Test compositional reasoning capability • Evaluate multi-sentence inference • Check for spurious correlations • Expect EM: 45-60% • Monitor generation quality (fluency vs. accuracy) |
|  | 70B | **Full Evaluation** • Open-Domain (rc.nocontext) • Reading Comp (rc.wikipedia) • Zero-shot, Few-shot, CoT | **Full Dataset** • 100% of train (138K) • Complete validation/test • All complexity levels • Web \+ Wikipedia evidence | **Fine-tuning Degradation** • SFT/RLHF safety tradeoffs • Instruction-following vs. accuracy • Distribution shift post-alignment • Comprehensive benchmark across all configs • Test both with/without evidence retrieval • Evaluate answer format following instructions • Expect EM: 60-75% (open-domain), 70-85% (with retrieval) • Monitor for alignment tax on factual accuracy • Test robustness to prompt variations |

**Tip:** 

### BBH (Big Bench Hard)

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| BBH (Big Bench Hard) | 1B | **Validation-Only (No Training)** • Answer-only prompts • 0-shot or 1-shot • Baseline capability assessment | **Foundational Tasks** • 5-8 easiest tasks • Binary/simple MCQ preferred • Avg input: 500-800 tokens **Tasks:** \- Boolean Expressions \- Sports Understanding \- Word Sorting \- Date Understanding \- Object Counting | **Basic Pattern Recognition** • Can model follow simple instructions? • Basic logical reasoning (AND/OR/NOT) • Lexical ordering • Simple world knowledge **Success Metric:** \>30% accuracy (vs random \~25%) **Sampling:** 50-100 examples/task **Evaluation Frequency:** Every 50B tokens **Red Flags:** \- Below random baseline \- Inability to parse instructions \- Inconsistent output format **Expected Performance:** 25-40% |
|  | 3B | **Validation \+ Light Probing** • Introduce 2-shot prompts • Answer-only (no CoT yet) • Task diversity assessment | **Intermediate Reasoning** • 12-15 tasks (add medium difficulty) • Mix of symbolic and semantic • Avg input: 800-1200 tokens **Add Tasks:** \- Causal Judgement \- Disambiguation QA \- Movie Recommendation \- Navigate \- Ruin Names \- Hyperbaton \- Reasoning About Colored Objects | **Multi-Step Inference** • Can model chain 2-3 reasoning steps? • Pronoun resolution • Spatial reasoning • Common sense causality • Semantic understanding **Success Metric:** \>40% accuracy **Sampling:** 100-150 examples/task **Evaluation Frequency:** Every 100B tokens **Red Flags:** \- Plateau on easy tasks \- Random performance on semantic tasks \- No improvement over 1B **Expected Performance:** 35-50% |
|  | 8B | **Full Diagnostic Testing** • 3-shot prompts • **Introduce CoT prompts** • Compare CoT vs answer-only • Emergent ability testing | **Advanced Reasoning** • 18-20 tasks (exclude hardest 3-5) • Multi-hop reasoning required • Avg input: 1200-1500 tokens **Add Tasks:** \- Logical Deduction (3-obj, 5-obj) \- Tracking Shuffled Objects (3-obj) \- Geometric Shapes \- Formal Fallacies \- Multi-Step Arithmetic \- Temporal Sequences \- Penguins in a Table | **Emergent Reasoning Capabilities** • Does CoT unlock performance gains? • Multi-hop logical deduction • Object/state tracking • Algorithmic reasoning • Geometric/spatial understanding **Success Metric:** \>50% with CoT (\>10% gain over answer-only) **Sampling:** 150-200 examples/task **Evaluation Frequency:** Every 150B tokens **Red Flags:** \- No CoT improvement (suggests reasoning deficit) \- Flat scaling curves \- Task-specific overfitting **Expected Performance:** 45-65% (with CoT) |
|  | 70B | **Comprehensive Benchmarking** • **Full 3-shot CoT** (standard) • All 23 tasks, full dataset • SFT/RL optimization target • Human-comparable evaluation | **All Tasks (Full Benchmark)** • All 23 tasks \+ subtasks (27 total) • Include hardest tasks • Full dataset per task • Avg input: 1500+ tokens **Add Final Tasks:** \- Logical Deduction (7-obj) \- Tracking Shuffled Objects (5-obj, 7-obj) \- Web of Lies \- Dyck Languages \- Salient Translation Error Detection \- Snarks | **Human-Level Reasoning** • Match/exceed human-rater performance • Complex multi-step reasoning • Robust CoT generation • Task generalization • Instruction following fidelity **Success Metric:** \>65% (human avg: 67.7%) **Target:** 70-85% **Sampling:** Full dataset (250/task, except 3 smaller) **Evaluation Frequency:** Every checkpoint during SFT/RL **Red Flags:** \- \<60% accuracy (below strong baselines) \- Degradation on easy tasks \- Invalid CoT reasoning **Expected Performance:** 65-85% **Post-SFT/RL Target:** 75-90% |

**Tip:** 

### IndicQA  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| IndicQA | 1B | Extractive QA only • Direct Inference • Zero-shot evaluation • Simple contexts (\< 300 tokens) | Low-Medium • Simple span extraction • Limited reasoning required | **Catastrophic Forgetting** • Loss of base capabilities during growth • Context understanding degradation **Language Interference** • Cross-lingual contamination • Script confusion (Devanagari vs Dravidian) **Overfitting** • Memorization vs understanding |
|  | 3B | Mixed Extractive \+ Abstractive • Direct Inference primary • 1-shot evaluation • Medium contexts (300-500 tokens) | **Medium** • Requires generation • Multi-step reasoning | **Language Disparity** • Performance gap between high/mid-resource languages • Uneven learning rates **Generation Quality** • Hallucination in abstractive tasks • Factual accuracy degradation **Scaling Instability** • Architecture adaptation stress • MoE routing inefficiency |
|  | 8B | Balanced Extractive \+ Abstractive • Both Direct Inference & Translate-Test • 3-5 shot evaluation • Complex contexts (500+ tokens) | **Medium-High** • Complex reasoning • Cultural nuance understanding • Multi-hop QA | **Low-Resource Language Failure** • Assamese, Odia, Punjabi underperformance • Translate-Test dependency **Domain Generalization** • Limited domain transfer • Cultural bias in answers **MoE Routing Issues** • Expert specialization problems • Load imbalance across experts |
|  | 70B | Full benchmark evaluation • Direct Inference \+ Translate-Test comparison • 5-10 shot evaluation • All context lengths • Instruction-tuned prompting | **High** • Maximum complexity • Open-domain reasoning • Cultural depth required | **Alignment Degradation** • RLHF/DPO misalignment with Indic contexts • Cultural insensitivity **Safety & Bias** • Language-specific harmful content generation • Cultural stereotyping **Instruction Following** • Prompt sensitivity in low-resource languages • Format compliance issues **Answer Quality** • Verbosity vs accuracy trade-off • Language-mixing in responses |

**Tip:** 

### ARC-Challenge  \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| ARC-Challenge | 1B | **Zero-shot Baseline** • No few-shot examples • Minimal prompting • Focus on task format understanding | **Low Complexity** • Single-hop questions • Direct factual recall • Simple vocabulary questions | **Catastrophic Failures** • Random guessing (25% baseline) • Format misunderstanding • Token position bias |
|  | 3B | **Few-shot Learning (5-shot)** • Introduce in-context examples • Basic chain-of-thought prompting • Pattern recognition focus | **Medium Complexity** • Two-hop reasoning • Simple causal relationships • Grade 3-6 level questions | **Memorization vs Understanding** • Overfitting to few-shot examples • Surface pattern matching • Position/formatting sensitivity |
|  | 8B | **Multi-shot Evaluation (25-shot)** • Standard benchmark protocol • Chain-of-thought reasoning • Comprehensive prompting | **High Complexity** • Multi-hop reasoning (3+ steps) • Commonsense inference • Grade 7-9 level questions | **Reasoning Shortcuts** • Statistical correlation exploitation • Keyword matching without understanding • Spurious pattern learning |
|  | 70B | • Full benchmark protocol • Instruction-following evaluation • Human-aligned prompting • Post-training optimization | **Maximum Complexity** • Complex multi-hop chains • Abstract reasoning • Integration of implicit knowledge • Edge cases and ambiguous questions | **Alignment & Generalization** • Overfitting to ARC during SFT • Catastrophic forgetting of base capabilities • Gaming the benchmark • Distribution shift failures |

**Tip:** 

### APPS (Automated Programming Progress Standard)\[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| APPS (Automated Programming Progress Standard) | 1B | **Zero-shot / Few-shot Evaluation** \- No fine-tuning on APPS training set \- Direct evaluation after pretraining \- Use 0-3 shot examples \- Focus on syntactic correctness | **Introductory Only** (1,000 problems) \- Simple algorithmic tasks \- Often one-line solutions \- Basic data structures \- String manipulation \- Simple math operations **Expected Pass@1: 2-5%** | **1\. Syntax Generation** \- Can model produce valid Python syntax? \- Basic indentation and structure \- Function definition understanding **2\. Simple Logic** \- One-step reasoning \- Direct input-output mapping **3\. Baseline Capability** \- Establish minimum viable performance |
|  | 3B | **Light Fine-tuning \+ Evaluation** \- Optional: Fine-tune on introductory subset only \- 1-2 epochs on easy problems \- Curriculum learning approach \- Monitor overfitting carefully | **Introductory (1,000) \+ Interview Subset (1,000-1,500)** \- Multi-step algorithms \- Basic data structures (arrays, hashmaps) \- Simple dynamic programming \- Two-pointer techniques **Expected Pass@1: 8-12%** | **1\. Multi-step Reasoning** \- Can model chain operations? \- Handle 2-3 step algorithms **2\. Data Structure Usage** \- Proper use of lists, dicts, sets \- Basic complexity awareness **3\. Edge Case Handling** \- Empty inputs, single elements \- Boundary conditions **4\. Memory Efficiency** \- Test if model scales approach appropriately |
|  | 8B | **Full Fine-tuning (No RL)** \- Fine-tune on full APPS training set \- 3-5 epochs with learning rate scheduling \- Mix difficulty levels in batches \- Implement early stopping | **Introductory (1,000) \+ Interview (2,000-2,500)** \- Advanced algorithms (BFS, DFS) \- Intermediate DP problems \- Graph traversal \- Greedy algorithms \- Sorting/searching variants **Expected Pass@1: 12-18%** | **1\. Algorithmic Complexity** \- Can the model select the right algorithm? \- O(n) vs O(n²) understanding **2\. Problem Decomposition** \- Breaking complex problems into steps \- Helper function generation **3\. Test Case Coverage** \- Does model think about edge cases? \- Corner case handling **4\. Code Organization** \- Proper function structure \- Variable naming clarity **5\. False Positive Rate** \- Test on problems with weak test coverage \- Verify solution generalization |
|  | 70B | **SFT \+ RL/RLHF Pipeline Phase 1 \- SFT:** \- Full APPS training set (5K problems) \- Include multiple solutions per problem \- 5-10 epochs with gradient accumulation **Phase 2 \- RL:** \- Use test case pass rate as reward \- PPO or similar policy optimization \- Self-debugging iterations \- Code execution feedback loop | **Full Test Set (5,000 problems)** \- Introductory: 1,000 \- Interview: 3,000 \- Competition: 1,000 **Includes:** \- Advanced DP (memoization, tabulation) \- Complex graph algorithms \- Number theory \- Computational geometry \- String algorithms (KMP, Z-algorithm) **Expected Pass@1: 20-30% Expected Pass@10: 35-45%** | **1\. Competition-Level Reasoning** \- Multi-step complex algorithms \- Problem-solving creativity \- Obscure edge cases **2\. Self-Correction Ability** \- Can a model debug its own code? \- Error recovery from failed tests **3\. Sample Efficiency** \- Performance vs number of attempts (k) \- Quality of top-k generations **4\. Generalization** \- Performance on temporal test split \- Robustness to problem variations **5\. RL Alignment** \- Reward hacking detection \- Test case overfitting \- Solution diversity vs correctness **6\. False Positive Mitigation** \- Cross-validate with additional tests \- Manual review of sample solutions |

**Tip:** 

### MATH \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| MATH | 1B | Foundation Pre-training \- Base mathematical reasoning \- Token-level next prediction \- No SFT at this stage | Level 1-2 \- Prealgebra (simple arithmetic) \- Basic Algebra \- Elementary Counting \~2,000-3,000 problems | Arithmetic Failure \- Basic calculation errors \- Single-step reasoning breakdown \- Format compliance (LaTeX boxing) |
|  | 3B | Intermediate Reasoning \- Enhanced multi-step logic \- Pattern recognition \- Light instruction tuning possible | Level 1-3 \- Intermediate Algebra \- Number Theory basics \- Probability fundamentals \~4,000-5,000 problems | Reasoning Chain Collapse \- Breaks down after 4-5 steps \- Symbolic manipulation errors \- Confusion between similar concepts |
|  | 8B | Advanced Problem Solving \- Complex reasoning emergence \- Better few-shot learning \- CoT capabilities solidify | Level 1-4 \- Advanced Algebra & Geometry \- Counting & Probability (complex) \- Early Precalculus \~8,000-10,000 problems | Domain-Specific Weakness \- Geometric visualization failure \- Complex combinatorics errors \- Inconsistent problem-type transfer |
|  | 70B | Full Capability \+ SFT/RLHF \- Complete mathematical reasoning \- Competition-level performance \- Solution quality optimization | Level 1-5 (Full) \- All subjects including hardest:   • Precalculus (Level 5\)   • Advanced Geometry   • Competition problems Full 12,500 problems | Contamination & Overfitting \- Memorization vs. true reasoning \- Brittleness to problem rephrasing \- Training data leakage \- Solution pattern copying |

**Tip:** 

### Indic-Bias (FairITales) \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| Indic-Bias (FairITales) | 1B | **Baseline Fairness Check** • Task: Plausibility only • Coverage: 20-30% templates • Identities: 2 types (Religion, Region) • Instances: \~4,000-6,000 | Low | **Overt Stereotyping**  • Blatant identity biases • Obvious discriminatory patterns • Basic stereotype association • Refusal rate patterns |
|  | 3B | **Systematic Bias Detection** • Tasks: Plausibility \+ Judgment • Coverage: 40-50% templates • Identities: All 4 types (Caste, Religion, Region, Tribe) • Instances: \~10,000-15,000 | Medium | **Allocative Harm**  • Systematic exclusion patterns • Favoring/disfavoring in decisions • Subtle discrimination in choices • Cross-identity consistency |
|  | 8B | **Comprehensive Fairness Audit** • Tasks: All 3 (Plausibility, Judgment, Generation) • Coverage: 65-75% templates • Identities: Full coverage \+ intersectional • Instances: \~15,000-20,000 | High | **Representational Harm**  • Stereotype reinforcement in generation • Nuanced bias in long-form text • Rationalization failures • Intersectional bias patterns |
|  | 70B | **Gold Standard Evaluation** • Tasks: All 3 with CoT variants • Coverage: 100% (all 20,000 templates) • Identities: Complete (85 groups) • Instances: \~25,000+ with all combinations | Very High | **Alignment Effectiveness**  • Post-training bias mitigation • Instruction-following fairness • Edge case handling • Real-world deployment readiness |

**Tip:** 

### HellaSwag \[Validate split (train/test) are there or not and decide to train\]

| Benchmark  | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| HellaSwag | 1B | Few-Shot (specifically 5-shot to 10-shot) MCQA (Multiple Choice Question Answering) using normalised log-likelihoods. | High | Sensitivity to Prompt Phrasing Dataset Noise and Errors Data Contamination |
|  | 3B | For models in the 3B range, the most reliable paradigm is Multiple Choice Question Answering (MCQA) using log-likelihood evaluation. | Medium | Memorization Trap Sensitivity to Variants Mitigation strategy ( advised to useHellaSwag Pro ) |
|  | 8B | Zero-Shot evaluation using Log-Likelihood. | Low to Medium | Paper Tiger Effect:A high score may represent memorisation rather than actual reasoning Lack of Nuance: At this size, the model may "max out" its ability to learn from this specific dataset. If you see a score of 85%, it doesn't necessarily mean the model is a "genius"; it just means it has successfully learned the narrow distribution of the HellaSwag dataset |
|  | 70B | Zero Shot ( Direct probability ) | Very Low | Overfitting and Evaluation Deception Benchmark Hardening: For a 70B model, it might be over-optimised for the specific linguistic style of HellaSwag while still failing at really world, novel reasoning tasks Memorisation: With the massive training budgets required for 70B models, the risk of the HellaSwag dataset leaking into the pre-training corpus is extremely high. |

### Winograde \[Validate split (train/test) are there or not and decide to train\]

| Benchmark | Model Size | Recommended Paradigm | Complexity Level | Primary Risk to Test |
| :---- | :---- | :---- | :---- | :---- |
| **WinoGrande** | **1B** | **Few-Shot (5-shot)**  Focus on basic pronoun resolution in simple physical contexts. | **Low** | **Grammar Reliance:** Failure to distinguish between nouns based on logic rather than sentence structure; random-level performance (50% baseline). |
| **WinoGrande** | **3B** | **Zero-Shot / MCQA**  Testing for emergent world knowledge without specific prompts. | **Medium** | **Statistical Biases:** The model may rely on "easy" word associations (e.g., "heavy" always goes with "suitcase") rather than actual spatial reasoning. |
| **WinoGrande** | **8B** | **Zero-Shot (Log-Likelihood)**  Standard evaluation to check for robust commonsense reasoning. | **Medium \- High** | **Brittleness:** High scores that collapse if a single word in the sentence is changed (perturbations), indicating pattern matching over true logic. |
| **WinoGrande** | **70B** | **Adversarial Evaluation**  Focusing on the "hard" subset (AfLite-filtered) to test frontier reasoning. | **Very High** | **Data Contamination:** High risk that the model has memorized the dataset during pre-training; scores above 85-90% may reflect memorization rather than reasoning. |

