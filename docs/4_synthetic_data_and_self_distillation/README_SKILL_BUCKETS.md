# Team 4: Skill Bucket Definitions

## 70B Mixture-of-Experts Language Model — Synthetic Data & Self-Distillation

---

## Document Scope

This document defines **38 skill buckets** for targeted synthetic data generation, diagnostic testing, and capability tracking across the training lifecycle of a 70B parameter Mixture-of-Experts (MoE) language model supporting English and Hindi (Devanagari + Romanized).

---

## Training Context

### Model Stages

| Stage | Parameters | Tokens | Primary Focus |
|-------|------------|--------|---------------|
| Stage 1 | 1B | 20B | Language fundamentals, grammar, basic patterns |
| Stage 2 | 3B | 40B | Knowledge acquisition, structured text understanding |
| Stage 3 | 8B | 100B | Reasoning emergence, code foundations |
| Stage 4 (MoE) | 16B active / 70B total | 240B | Advanced reasoning, specialization, alignment |

### Difficulty Bands

| Band | Level | Description | CoT Policy |
|------|-------|-------------|------------|
| **B0** | Nursery | Grammar, syntax, high-frequency constructions | No CoT |
| **B1** | Primary | Fluent everyday language, common knowledge | No CoT |
| **B2** | High School | Structured knowledge, implicit reasoning | No CoT |
| **B3** | Undergraduate | Multi-step explanations, meaningful technical content | CoT rare, capped, short |
| **B4** | Graduate | Math, algorithms, proofs, controlled CoT | CoT allowed, capped |
| **B5** | PhD | Maximum complexity, advanced reasoning, limited agentic | CoT allowed, never dominant |

### Phase Tags

| Tag | Meaning | Description |
|-----|---------|-------------|
| `[PRE]` | Pretraining | Core capability built during pretraining |
| `[SFT]` | Supervised Fine-Tuning | Capability refined during SFT |
| `[RLHF]` | Reinforcement Learning | Capability aligned during RLHF |
| `[ALL]` | All Phases | Relevant across entire training lifecycle |

---

## Skill Bucket Taxonomy Overview

### Summary Statistics

- **Total Skill Buckets**: 38
- **Evaluation Planes**: 6
- **Critical Buckets**: 18 (marked with ✓)
- **Hindi-Specific Buckets**: 6

---

## Master Skill Bucket Table

| # | Bucket ID | Bucket Name | Plane | Phase | Bands | Critical | Primary Benchmarks |
|---|-----------|-------------|-------|-------|-------|----------|-------------------|
| 1 | FND-LEX-EN | English Lexical & Syntactic | Foundation | `[PRE]` | B0-B2 | ✓ | HellaSwag |
| 2 | FND-LEX-HI | Hindi Lexical & Syntactic | Foundation | `[PRE]` | B0-B2 | ✓ | HellaSwag-Hi |
| 3 | FND-SEM | Semantic Understanding | Foundation | `[PRE]` | B1-B3 | | WinoGrande, COPA |
| 4 | FND-DIS | Discourse & Coherence | Foundation | `[PRE]` | B2-B3 | | WSC |
| 5 | FND-LCX | Long-Context Retention | Foundation | `[PRE]` | B3-B5 | ✓ | LongBench |
| 6 | FND-FACT | Factual Knowledge Encoding | Foundation | `[PRE]` | B1-B3 | | TriviaQA, NQ |
| 7 | RSN-LOG | Logical Deduction | Reasoning | `[PRE]` `[SFT]` | B3-B5 | ✓ | BigBench-Logic |
| 8 | RSN-CAUS | Causal Reasoning | Reasoning | `[PRE]` `[SFT]` | B3-B5 | | COPA |
| 9 | RSN-ANAL | Analogical Reasoning | Reasoning | `[PRE]` `[SFT]` | B3-B4 | | BigBench-Analogy |
| 10 | RSN-ARITH | Arithmetic & Numerical | Reasoning | `[PRE]` `[SFT]` | B2-B4 | ✓ | GSM8K (subset) |
| 11 | RSN-ALG | Algebraic & Symbolic | Reasoning | `[PRE]` `[SFT]` | B3-B5 | | MATH-Algebra |
| 12 | RSN-WPT | Word Problem Translation | Reasoning | `[PRE]` `[SFT]` | B3-B5 | ✓ | GSM8K |
| 13 | RSN-ADVMATH | Advanced Mathematical | Reasoning | `[SFT]` `[RLHF]` | B4-B5 | | MATH L4-5 |
| 14 | RSN-MATH-HI | Mathematical Reasoning (Hindi) | Reasoning | `[SFT]` | B3-B5 | ✓ | GSM8K-Hi |
| 15 | RSN-CS | Commonsense Reasoning | Reasoning | `[PRE]` `[SFT]` | B2-B4 | | HellaSwag, PIQA, ARC |
| 16 | RSN-MH | Multi-Hop Reasoning | Reasoning | `[SFT]` `[RLHF]` | B4-B5 | | HotpotQA |
| 17 | CODE-SYN | Code Syntax & Structure | Code | `[PRE]` | B2-B3 | ✓ | Internal |
| 18 | CODE-COMP | Code Comprehension | Code | `[PRE]` `[SFT]` | B3-B4 | ✓ | MBPP (subset) |
| 19 | CODE-GEN-T1 | Code Generation (Python, JS) | Code | `[PRE]` `[SFT]` | B3-B5 | ✓ | HumanEval, MBPP |
| 20 | CODE-GEN-T2 | Code Generation (Java, C++, TS, Go) | Code | `[SFT]` | B3-B5 | | MultiPL-E |
| 21 | CODE-GEN-T3 | Code Generation (Rust, SQL, Bash) | Code | `[SFT]` | B3-B5 | | Spider, MultiPL-E |
| 22 | CODE-DBG | Code Debugging & Error Correction | Code | `[SFT]` `[RLHF]` | B3-B5 | ✓ | DebugBench |
| 23 | CODE-OPT | Code Optimization & Refactoring | Code | `[SFT]` `[RLHF]` | B4-B5 | | Internal |
| 24 | CODE-TEST | Test Generation & Verification | Code | `[SFT]` `[RLHF]` | B3-B5 | | EvalPlus |
| 25 | LANG-HI-COMP | Hindi Comprehension | Language | `[PRE]` `[SFT]` | B1-B4 | ✓ | MMLU-Hi |
| 26 | LANG-HI-GEN | Hindi Generation | Language | `[PRE]` `[SFT]` | B1-B4 | | Internal |
| 27 | LANG-TRANS | English-Hindi Translation | Language | `[SFT]` | B2-B4 | | FLORES-200 |
| 28 | LANG-MIX | Code-Mixed Language (Hinglish) | Language | `[PRE]` `[SFT]` | B2-B4 | | GLUECoS |
| 29 | LANG-HI-LOG | Hindi Logical Reasoning | Language | `[SFT]` | B3-B5 | | BigBench-Hi |
| 30 | ALN-INST | Instruction Following | Alignment | `[SFT]` `[RLHF]` | B2-B5 | ✓ | IFEval |
| 31 | ALN-STRUCT | Structured Output Generation | Alignment | `[SFT]` `[RLHF]` | B3-B5 | ✓ | IFEval (format) |
| 32 | ALN-HALL | Hallucination Resistance | Alignment | `[SFT]` `[RLHF]` | B2-B5 | ✓ | TruthfulQA |
| 33 | ALN-SAFE | Safety & Harm Avoidance | Alignment | `[SFT]` `[RLHF]` | B0-B5 | ✓ | Red-team suite |
| 34 | ALN-HELP | Helpfulness & Engagement | Alignment | `[SFT]` `[RLHF]` | B1-B5 | | MT-Bench |
| 35 | PRD-ROB | Input Robustness | Production | `[ALL]` | B1-B5 | | CheckList |
| 36 | PRD-SUM | Summarization | Production | `[SFT]` | B2-B4 | | CNN/DM, XSum |
| 37 | PRD-IE | Information Extraction | Production | `[SFT]` | B2-B4 | | CoNLL, TACRED |
| 38 | PRD-CREAT | Creative Writing | Production | `[SFT]` `[RLHF]` | B2-B5 | | Human Eval |

---

## Plane Distribution Summary

| Plane | Bucket Count | Critical Count | Bucket IDs |
|-------|--------------|----------------|------------|
| **Plane 1: Foundation** | 6 | 3 | FND-LEX-EN, FND-LEX-HI, FND-SEM, FND-DIS, FND-LCX, FND-FACT |
| **Plane 2: Reasoning** | 10 | 4 | RSN-LOG, RSN-CAUS, RSN-ANAL, RSN-ARITH, RSN-ALG, RSN-WPT, RSN-ADVMATH, RSN-MATH-HI, RSN-CS, RSN-MH |
| **Plane 3: Code** | 8 | 4 | CODE-SYN, CODE-COMP, CODE-GEN-T1, CODE-GEN-T2, CODE-GEN-T3, CODE-DBG, CODE-OPT, CODE-TEST |
| **Plane 4: Language** | 5 | 1 | LANG-HI-COMP, LANG-HI-GEN, LANG-TRANS, LANG-MIX, LANG-HI-LOG |
| **Plane 5: Alignment** | 5 | 4 | ALN-INST, ALN-STRUCT, ALN-HALL, ALN-SAFE, ALN-HELP |
| **Plane 6: Production** | 4 | 0 | PRD-ROB, PRD-SUM, PRD-IE, PRD-CREAT |

---

## Acceptance Thresholds Summary

| Bucket ID | Key Metric | Target | SOTA Reference |
|-----------|-----------|--------|----------------|
| FND-LEX-EN | Grammar validity | ≥ 98% | GPT-4: ~99% |
| FND-LEX-HI | Hindi accuracy | ≥ 95% | GPT-4: ~94% |
| FND-SEM | WinoGrande | ≥ 75% | GPT-4: ~87% |
| FND-LCX | Retrieval @ 32K | ≥ 85% | Claude 3: ~90% |
| FND-FACT | Closed-book QA | ≥ 70% | GPT-4: ~80% |
| RSN-LOG | Contradiction rate | < 5% | GPT-4: ~88% accuracy |
| RSN-ARITH | Single-step accuracy | ≥ 95% | GPT-4: ~95% |
| RSN-WPT | GSM8K | ≥ 75% | GPT-4: ~92% |
| RSN-ADVMATH | MATH L5 | ≥ 30% | GPT-4: ~52% overall |
| CODE-GEN-T1 | HumanEval Pass@1 | ≥ 70% | GPT-4: ~85% |
| CODE-DBG | Bug localization | ≥ 80% | GPT-4: ~75% |
| LANG-HI-COMP | Delta vs English | ≤ 5% | GPT-4: ~5-8% gap |
| LANG-TRANS | BLEU (En→Hi) | ≥ 35 | Google: ~38 |
| ALN-INST | IFEval strict | ≥ 70% | GPT-4: ~80% |
| ALN-STRUCT | JSON validity | ≥ 98% | Claude 3: ~98% |
| ALN-HALL | Hallucination rate | < 10% | Claude 3: ~8% |
| ALN-SAFE | Harmful generation | < 1% | Claude 3: <1% |
| PRD-SUM | ROUGE-L | ≥ 35 | GPT-4: ~40 |

---

## Detailed Skill Bucket Specifications

---

# Plane 1: Foundation Modeling

Foundation buckets establish core language competence required before higher-order reasoning can emerge.

---

## Bucket 1: Lexical & Syntactic Competence (English)

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-LEX-EN` |
| **Phase** | `[PRE]` |
| **Bands** | B0-B2 |
| **Critical** | ✓ Yes |

### Description
Core English grammar, syntax, morphology, and sentence structure. This bucket ensures the model produces grammatically valid English text with proper sentence construction.

### Sub-skills
- Subject-verb agreement
- Tense consistency and aspect
- Pronoun resolution and agreement
- Clause structure (subordinate, relative, conditional)
- Punctuation and formatting conventions
- Morphological inflection

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| HellaSwag (completion) | Accuracy | ≥ 85% | GPT-4: ~95% |
| Internal grammar probes | Error rate | < 2% | — |

### Acceptance Criteria
- Grammar error rate < 2%
- Syntactic validity > 98%
- No systematic agreement failures

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Subject-verb disagreement | Plural subject with singular verb | "The group of students *are* ready" → should be "is" |
| Dangling modifiers | Modifier attached to wrong element | "Running quickly, the finish line was crossed" |
| Tense inconsistency | Mixed tenses within passage | "He walked to the store and *buys* milk" |
| Run-on sentences | Missing proper sentence boundaries | Comma splices, fused sentences |

### Synthetic Data Priorities
- Grammar correction pairs (error → correct)
- Sentence completion with agreement constraints
- Complex clause construction examples

---

## Bucket 2: Lexical & Syntactic Competence (Hindi - Devanagari)

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-LEX-HI` |
| **Phase** | `[PRE]` |
| **Bands** | B0-B2 |
| **Critical** | ✓ Yes |

### Description
Hindi grammar in Devanagari script, including morphological agreement, postposition usage, and verb conjugation. Ensures the model handles Hindi's SOV structure and gender-number agreement correctly.

### Sub-skills
- Gender-number agreement (विभक्ति / vibhakti)
- Postposition usage (में, पर, को, से, etc.)
- Verb conjugation patterns (tense, aspect, mood)
- Honorific forms (आप / तुम / तू distinctions)
- Sandhi rules (vowel/consonant combination)
- Compound verb constructions (मिलकर verb pairs)

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| HellaSwag-Hi | Accuracy | ≥ 80% | GPT-4: ~85% |
| Internal Hindi probes | Grammar accuracy | ≥ 95% | — |
| — | Delta vs English | ≤ 5% | GPT-4: ~5-8% gap |

### Acceptance Criteria
- Hindi grammatical accuracy ≥ 95%
- Gender agreement accuracy ≥ 98%
- Honorific consistency ≥ 95%
- Performance delta vs English ≤ 5%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Gender agreement error | Masculine noun with feminine verb | "लड़का स्कूल *जाती* है" → should be "जाता" |
| Postposition case error | Wrong postposition for semantic role | "मैं किताब *में* पढ़ता हूं" → should be "को" |
| Honorific inconsistency | Mixing formal/informal in same context | "आप कहाँ *जाता* है?" → should be "जाते हैं" |
| Matra errors | Incorrect vowel diacritics | "किताब" vs "कीताब" |

### Synthetic Data Priorities
- Gender agreement exercises (masculine/feminine/plural)
- Postposition drills with semantic roles
- Honorific consistency passages
- Devanagari script validation pairs

---

## Bucket 3: Semantic Understanding

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-SEM` |
| **Phase** | `[PRE]` |
| **Bands** | B1-B3 |
| **Critical** | No |

### Description
Understanding word meaning in context, semantic relationships, and meaning composition. Enables correct interpretation beyond surface syntax.

### Sub-skills
- Word sense disambiguation
- Metaphor and idiom interpretation
- Semantic role labeling (agent, patient, instrument)
- Entailment recognition
- Paraphrase detection
- Antonymy and synonymy

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| WinoGrande | Accuracy | ≥ 75% | GPT-4: ~87% |
| COPA | Accuracy | ≥ 85% | GPT-4: ~95% |
| Internal semantic probes | Entailment F1 | ≥ 85% | — |

### Acceptance Criteria
- WinoGrande accuracy ≥ 75%
- Entailment detection F1 ≥ 85%
- Idiom interpretation accuracy ≥ 80%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Literal interpretation | Missing figurative meaning | "He kicked the bucket" → physical kicking |
| Context insensitivity | Ignoring disambiguating context | "Bank" as financial vs river bank |
| False entailment | Surface similarity causing incorrect inference | "Dog bit man" ≠ "Man bit dog" |
| Semantic role confusion | Swapping agent and patient | Misattributing who did what |

### Synthetic Data Priorities
- Word sense disambiguation examples
- Metaphor interpretation pairs
- Entailment/contradiction/neutral triplets

---

## Bucket 4: Discourse & Coherence

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-DIS` |
| **Phase** | `[PRE]` |
| **Bands** | B2-B3 |
| **Critical** | No |

### Description
Multi-sentence coherence, discourse relations, and narrative flow. Ensures generated text maintains logical and topical consistency across paragraphs.

### Sub-skills
- Coreference resolution (pronouns, definite descriptions)
- Discourse connective understanding (however, therefore, because)
- Topic continuity and progression
- Rhetorical structure recognition
- Paragraph organization
- Given-new information structure

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| WSC (Winograd Schema) | Accuracy | ≥ 85% | GPT-4: ~95% |
| Internal coherence probes | Coherence score | ≥ 90% | — |
| — | Coreference accuracy | ≥ 85% | — |

### Acceptance Criteria
- Coreference resolution accuracy ≥ 85%
- Discourse coherence score ≥ 90%
- No contradictions across paragraphs

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Pronoun ambiguity | Unclear antecedent selection | "John told Bob that he was wrong" — who is "he"? |
| Topic drift | Losing main topic over long generation | Starting about X, ending about unrelated Y |
| Contradiction | Asserting P then later ¬P | "It was raining" ... "The sunny day continued" |
| Broken discourse | Misusing connectives | "Therefore" without logical premise |

### Synthetic Data Priorities
- Coreference resolution exercises
- Paragraph ordering tasks
- Discourse connective fill-in

---

## Bucket 5: Long-Context Retention (>32K)

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-LCX` |
| **Phase** | `[PRE]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Information retention and retrieval across extended contexts (32K+ tokens). Critical for document-level tasks, multi-turn conversations, and complex reasoning over long inputs.

### Sub-skills
- Key information extraction at varying positions
- Cross-document reasoning
- Context boundary awareness
- Attention distribution efficiency
- Position-independent retrieval

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| LongBench | Aggregate score | ≥ 70% | GPT-4 Turbo: ~75% |
| Internal needle retrieval | Accuracy @ 32K | ≥ 85% | Claude 3: ~90% |
| Multi-document QA | F1 | ≥ 70% | — |

### Acceptance Criteria
- Retrieval accuracy ≥ 85% at 32K tokens
- No catastrophic decay after 24K tokens
- Position bias < 10% (beginning vs middle vs end)

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Lost in the middle | Poor retrieval from mid-context | Needle at position 50% missed |
| Recency bias | Over-weighting recent tokens | Ignoring early context facts |
| Primacy bias | Over-weighting initial tokens | Missing late context updates |
| Context truncation | Acting as if context is shorter | Answering based on partial context |

### Synthetic Data Priorities
- Position-varied fact retrieval
- Multi-document synthesis
- Long-range dependency resolution

---

## Bucket 6: Factual Knowledge Encoding

| Attribute | Value |
|-----------|-------|
| **ID** | `FND-FACT` |
| **Phase** | `[PRE]` |
| **Bands** | B1-B3 |
| **Critical** | No |

### Description
Storage and retrieval of world knowledge including entities, facts, relationships, and events. Enables accurate closed-book question answering.

### Sub-skills
- Named entity recognition
- Factual recall (dates, places, events, figures)
- Relationship knowledge (X is Y of Z)
- Temporal knowledge ordering
- Quantitative facts (populations, distances, etc.)

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| TriviaQA (closed-book) | Accuracy | ≥ 70% | GPT-4: ~80% |
| NaturalQuestions | Accuracy | ≥ 45% | GPT-4: ~55% |
| Internal fact probes | Entity F1 | ≥ 85% | — |

### Acceptance Criteria
- Factual accuracy ≥ 70% on closed-book QA
- Entity recognition F1 ≥ 85%
- Relationship accuracy ≥ 75%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Entity confusion | Mixing similar entities | Confusing John Adams with John Quincy Adams |
| Temporal anachronism | Wrong time attribution | "Einstein discovered relativity in 1850" |
| Quantitative hallucination | Fabricating numbers | Made-up population statistics |
| False confidence | High certainty on wrong facts | Asserting incorrect capitals |

### Synthetic Data Priorities
- Entity disambiguation pairs
- Temporal ordering exercises
- Fact verification examples (true/false)

---

# Plane 2: Reasoning & Intelligence

Reasoning buckets enable logical inference, mathematical problem-solving, and multi-step deduction.

---

## Bucket 7: Logical Deduction

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-LOG` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Formal logical reasoning including syllogisms, propositional logic, and deductive inference. Core capability for reliable reasoning chains.

### Sub-skills
- Modus ponens (if P then Q; P; therefore Q)
- Modus tollens (if P then Q; not Q; therefore not P)
- Syllogistic reasoning (categorical logic)
- Propositional logic (and, or, not, implies)
- Predicate logic (basic quantifiers)
- Logical equivalence and contradiction detection

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| BigBench (logic subset) | Accuracy | ≥ 80% | GPT-4: ~88% |
| FOLIO | Accuracy | ≥ 70% | GPT-4: ~75% |
| Internal logic probes | Contradiction rate | < 5% | — |

### Acceptance Criteria
- Logical contradiction rate < 5%
- Syllogism accuracy ≥ 85%
- Valid inference rate ≥ 90%
- Modus ponens/tollens accuracy ≥ 95%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Affirming the consequent | Invalid: If P→Q and Q, concluding P | "If rain then wet; wet; therefore rain" |
| Denying the antecedent | Invalid: If P→Q and ¬P, concluding ¬Q | "If rain then wet; no rain; therefore not wet" |
| Contrapositive failure | Not applying ¬Q→¬P correctly | Missing valid inferences |
| Universal overgeneralization | "All X are Y" from limited cases | Hasty generalization |

### Synthetic Data Priorities
- Explicit logical form exercises
- Fallacy identification (negative examples)
- Multi-step deduction chains
- Contradiction detection pairs

---

## Bucket 8: Causal Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-CAUS` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Understanding cause-effect relationships, counterfactual reasoning, and causal chain analysis.

### Sub-skills
- Cause-effect identification
- Counterfactual reasoning ("What if X had happened?")
- Temporal causation (cause precedes effect)
- Confounding variable awareness
- Causal chain reasoning (A→B→C)
- Intervention vs observation distinction

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| COPA | Accuracy | ≥ 90% | GPT-4: ~95% |
| BigBench (causal) | Accuracy | ≥ 75% | GPT-4: ~80% |
| Internal causal probes | Counterfactual consistency | ≥ 85% | — |

### Acceptance Criteria
- Causal accuracy ≥ 80%
- Counterfactual consistency ≥ 85%
- No correlation-causation conflation in explicit tests

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Correlation→causation | Inferring cause from co-occurrence | "Ice cream sales and drowning correlate, so..." |
| Reverse causation | Getting cause/effect backwards | "Fever causes infection" |
| Ignoring confounders | Missing third variable | Not considering hidden common causes |
| Temporal violation | Effect before cause | Logically impossible orderings |

### Synthetic Data Priorities
- Confounded scenario analysis
- Counterfactual reasoning pairs
- Causal chain completion

---

## Bucket 9: Analogical Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-ANAL` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B4 |
| **Critical** | No |

### Description
Pattern recognition, analogy completion, and relational mapping across domains.

### Sub-skills
- Proportional analogies (A:B :: C:D)
- Structural mapping across domains
- Cross-domain transfer
- Pattern extrapolation
- Relational similarity detection

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| BigBench (analogy) | Accuracy | ≥ 75% | GPT-4: ~82% |
| Internal analogy probes | Accuracy | ≥ 70% | — |
| Cross-domain transfer | Success rate | ≥ 65% | — |

### Acceptance Criteria
- Analogy completion accuracy ≥ 75%
- Cross-domain transfer ≥ 70%
- Structural (not surface) matching ≥ 80%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Surface matching | Ignoring deep structural similarity | Matching by word similarity not relation |
| Overgeneralization | Extending analogy beyond validity | Pushing metaphor too far |
| Domain specificity | Failing to transfer to new domains | Can't apply learned pattern elsewhere |

### Synthetic Data Priorities
- Cross-domain analogy pairs
- Structural vs surface discrimination
- Novel analogy construction

---

## Bucket 10: Arithmetic & Numerical Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-ARITH` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | ✓ Yes |

### Description
Basic arithmetic operations, numerical manipulation, and quantitative estimation.

### Sub-skills
- Four operations (+, −, ×, ÷)
- Order of operations (PEMDAS/BODMAS)
- Fractions and decimals
- Percentages and ratios
- Numerical estimation
- Unit conversion

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| GSM8K (arithmetic subset) | Accuracy | ≥ 90% | GPT-4: ~95% |
| MATH (Level 1-2) | Accuracy | ≥ 85% | GPT-4: ~90% |
| Internal arithmetic probes | Single-step | ≥ 95% | — |
| — | Multi-step | ≥ 85% | — |

### Acceptance Criteria
- Single-operation accuracy ≥ 95%
- Multi-step arithmetic ≥ 85%
- Estimation within 20% of true value
- No systematic carrying/borrowing errors

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Carrying errors | Mistakes in multi-digit addition | 47 + 38 = 75 (should be 85) |
| Order of operations | PEMDAS violations | 2 + 3 × 4 = 20 (should be 14) |
| Decimal placement | Wrong decimal position | 1.5 × 2 = 30 (should be 3.0) |
| Large number errors | Failures with big numbers | Errors increase with digit count |

### Synthetic Data Priorities
- Multi-digit operation drills
- Order of operations exercises
- Decimal and fraction arithmetic
- Estimation tasks

---

## Bucket 11: Algebraic & Symbolic Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-ALG` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Variable manipulation, equation solving, and symbolic mathematics.

### Sub-skills
- Linear equation solving
- Quadratic equations (factoring, formula)
- Systems of equations
- Symbolic simplification
- Variable substitution
- Inequality reasoning

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MATH (Algebra) | Accuracy | ≥ 65% | GPT-4: ~70% |
| GSM8K (algebraic) | Accuracy | ≥ 80% | GPT-4: ~90% |
| Internal algebra probes | Linear equation | ≥ 90% | — |
| — | Quadratic | ≥ 80% | — |

### Acceptance Criteria
- Linear equation accuracy ≥ 90%
- Quadratic equation accuracy ≥ 80%
- Systems of equations ≥ 75%
- Symbolic simplification ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Sign errors | Mistakes during manipulation | −(x − 3) = −x − 3 (should be −x + 3) |
| Incorrect factoring | Wrong factor identification | x² + 5x + 6 = (x+1)(x+6) (should be (x+2)(x+3)) |
| Distribution errors | Failing to distribute correctly | 2(x + 3) = 2x + 3 (should be 2x + 6) |
| Extraneous solutions | Including invalid solutions | Not checking domain restrictions |

### Synthetic Data Priorities
- Step-by-step equation solving
- Factoring exercises
- Systems of equations
- Variable substitution chains

---

## Bucket 12: Word Problem Translation

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-WPT` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Converting natural language problem descriptions into mathematical formulations.

### Sub-skills
- Entity extraction (quantities, units, rates)
- Relationship identification (more than, times as much)
- Constraint formulation
- Multi-step problem decomposition
- Irrelevant information filtering
- Question identification

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| GSM8K | Accuracy | ≥ 75% | GPT-4: ~92% |
| MATH (word problems) | Accuracy | ≥ 60% | GPT-4: ~70% |
| SVAMP | Accuracy | ≥ 80% | GPT-4: ~90% |

### Acceptance Criteria
- GSM8K accuracy ≥ 75%
- Problem decomposition accuracy ≥ 80%
- Irrelevant information filtering ≥ 90%
- Correct question identification ≥ 95%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Wrong question | Solving for wrong unknown | Calculating total when asked for difference |
| Unit confusion | Mixing or ignoring units | Adding hours and minutes directly |
| Irrelevant inclusion | Using distractor information | Red herring numbers included |
| Missing constraints | Overlooking implicit requirements | Not considering "at least" or "positive" |

### Synthetic Data Priorities
- Multi-step word problems with decomposition
- Distractor-heavy problems
- Unit conversion word problems
- Constraint-explicit problems

---

## Bucket 13: Advanced Mathematical Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-ADVMATH` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B4-B5 |
| **Critical** | No |

### Description
Higher mathematics including calculus, probability, combinatorics, and proof techniques.

### Sub-skills
- Differentiation and integration
- Probability and statistics
- Combinatorics (permutations, combinations)
- Basic proof techniques (direct, contradiction, induction)
- Mathematical induction
- Series and sequences

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MATH (Level 4) | Accuracy | ≥ 50% | GPT-4: ~55% |
| MATH (Level 5) | Accuracy | ≥ 30% | GPT-4: ~35% |
| Internal proof probes | Validity | ≥ 70% | — |

### Acceptance Criteria
- MATH Level 4 ≥ 50%
- MATH Level 5 ≥ 30%
- Proof validity ≥ 70%
- Probability calculation ≥ 75%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Differentiation rules | Wrong derivative rules | d/dx(sin x) = sin x (should be cos x) |
| Probability confusion | Mixing distributions | Using normal formula for binomial |
| Invalid proof steps | Logical gaps in proofs | Assuming what needs to be proved |
| Combinatorial errors | Over/undercounting | Missing or double-counting cases |

### Synthetic Data Priorities
- Calculus problem-solution pairs
- Probability scenario analysis
- Proof construction with explicit steps
- Combinatorics with verification

---

## Bucket 14: Mathematical Reasoning (Hindi)

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-MATH-HI` |
| **Phase** | `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Mathematical problem solving presented in Hindi (Devanagari script).

### Sub-skills
- Hindi numeral understanding (१, २, ३, ४, ५, ६, ७, ८, ९, ०)
- Mathematical terminology in Hindi (जोड़, घटाव, गुणा, भाग)
- Word problem translation (Hindi → Math)
- Hindi number words (एक सौ बीस = 120)
- Mixed numeral handling (Devanagari + Arabic)

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| GSM8K-Hi (translated) | Accuracy | ≥ 70% | Limited data |
| Internal Hindi math | Delta vs English | ≤ 8% | — |
| Numeral parsing | Accuracy | 100% | — |

### Acceptance Criteria
- Hindi math accuracy within 8% of English equivalent
- Hindi numeral parsing 100%
- Mathematical term understanding ≥ 95%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Numeral confusion | Mixing Devanagari digits | ६ (6) read as ९ (9) |
| Term mistranslation | Wrong operation from Hindi term | "गुणा" (multiply) as addition |
| Number word errors | Wrong parsing of spelled numbers | "इक्कीस" (21) as 12 |
| Unit confusion | Hindi units mishandled | "दर्जन" (dozen) not converted |

### Synthetic Data Priorities
- Hindi arithmetic with Devanagari numerals
- Mathematical terminology mapping
- Number word to digit conversion
- Hindi word problems

---

## Bucket 15: Commonsense Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-CS` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | No |

### Description
Everyday reasoning about the physical and social world, including intuitive physics, social understanding, and temporal reasoning.

### Sub-skills
- Physical commonsense (objects, gravity, fluids)
- Social commonsense (intentions, reactions, norms)
- Temporal commonsense (event ordering, duration)
- Spatial reasoning (containment, relative position)
- Functional knowledge (what objects are for)

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| HellaSwag | Accuracy | ≥ 85% | GPT-4: ~95% |
| PIQA | Accuracy | ≥ 80% | GPT-4: ~90% |
| ARC-Challenge | Accuracy | ≥ 75% | GPT-4: ~96% |
| Social IQa | Accuracy | ≥ 75% | GPT-4: ~85% |

### Acceptance Criteria
- HellaSwag ≥ 85%
- PIQA ≥ 80%
- ARC-Challenge ≥ 75%
- No physical impossibilities in generation

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Physical violations | Ignoring physics | "The ball fell up" |
| Social implausibility | Unlikely social inferences | "He was happy to lose his job" |
| Temporal impossibility | Events in wrong order | Effect before cause |
| Functional errors | Wrong object usage | "Cut bread with a spoon" |

### Synthetic Data Priorities
- Physical scenario completion
- Social situation reasoning
- Temporal ordering tasks
- Object function descriptions

---

## Bucket 16: Multi-Hop Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `RSN-MH` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B4-B5 |
| **Critical** | No |

### Description
Chaining multiple reasoning steps, aggregating evidence, and tracking intermediate conclusions.

### Sub-skills
- Evidence aggregation across sources
- Intermediate conclusion tracking
- Reasoning chain construction
- Dead-end detection and backtracking
- Bridge entity identification

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| HotpotQA | F1 | ≥ 65% | GPT-4: ~70% |
| MuSiQue | Accuracy | ≥ 50% | GPT-4: ~55% |
| StrategyQA | Accuracy | ≥ 70% | GPT-4: ~80% |

### Acceptance Criteria
- Multi-hop accuracy ≥ 65%
- Reasoning chain validity ≥ 80%
- No hallucinated intermediate steps
- Correct bridge entity identification ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Shortcut reasoning | Skipping necessary steps | Jumping to conclusion without evidence |
| Error propagation | Early error compounds | Wrong intermediate → wrong final |
| Fabricated evidence | Hallucinating bridge facts | Making up connecting information |
| Goal drift | Losing track of question | Answering different question |

### Synthetic Data Priorities
- Explicit multi-hop chains with intermediate steps
- Bridge entity identification
- Reasoning path verification

---

# Plane 3: Code Capabilities

Code buckets cover understanding, generation, debugging, and testing across multiple programming languages.

---

## Bucket 17: Code Syntax & Structure

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-SYN` |
| **Phase** | `[PRE]` |
| **Bands** | B2-B3 |
| **Critical** | ✓ Yes |

### Description
Understanding programming language syntax across languages, including structure, formatting, and language identification.

### Sub-skills
- Syntax validity recognition
- Language identification
- Indentation and structure
- Comment parsing
- Import/include understanding
- Bracket/delimiter matching

### Supported Languages
**Tier 1**: Python, JavaScript  
**Tier 2**: Java, C++, TypeScript, Go  
**Tier 3**: Rust, SQL, Bash, Ruby, PHP, Swift, Kotlin

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| Internal syntax validation | Error detection | ≥ 95% | GPT-4: ~98% |
| Language identification | Accuracy | ≥ 98% | GPT-4: ~99% |
| Valid generation | Syntax validity | ≥ 98% | GPT-4: ~99% |

### Acceptance Criteria
- Syntax error detection ≥ 95%
- Language identification ≥ 98%
- Valid syntax generation ≥ 98%
- Indentation correctness (Python) ≥ 99%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Bracket mismatch | Unbalanced delimiters | Missing closing brace |
| Indentation errors | Python whitespace issues | Mixed tabs/spaces |
| Semicolon issues | C-family punctuation | Missing or extra semicolons |
| Language confusion | Mixing syntax rules | Python syntax in JavaScript |

### Synthetic Data Priorities
- Syntax error detection pairs
- Language identification examples
- Bracket-heavy code completion

---

## Bucket 18: Code Comprehension

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-COMP` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B4 |
| **Critical** | ✓ Yes |

### Description
Understanding code functionality, tracing execution, and explaining code behavior.

### Sub-skills
- Function purpose identification
- Variable state tracking
- Control flow understanding
- Side effect recognition
- Complexity assessment
- Code summarization

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MBPP (comprehension) | Accuracy | ≥ 80% | GPT-4: ~90% |
| CodeXGLUE (summarization) | BLEU | ≥ 30 | GPT-4: ~35 |
| Internal explanation | Accuracy | ≥ 85% | — |

### Acceptance Criteria
- Code explanation accuracy ≥ 85%
- Function purpose identification ≥ 90%
- Execution trace accuracy ≥ 85%
- Side effect identification ≥ 80%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Missing edge cases | Not identifying boundary behavior | Off-by-one conditions |
| Loop miscounting | Wrong iteration count | range(5) as 5 iterations starting at 1 |
| Side effect blindness | Missing state changes | Not noticing global modification |
| Recursion confusion | Wrong base case analysis | Misunderstanding termination |

### Synthetic Data Priorities
- Code explanation pairs
- Execution tracing exercises
- Side effect identification
- Complexity analysis

---

## Bucket 19: Code Generation (Tier 1: Python, JavaScript)

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-GEN-T1` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Generating correct, functional code from natural language specifications in Python and JavaScript.

### Sub-skills
- Function implementation from docstring
- Algorithm translation
- API usage (standard libraries)
- Error handling
- Edge case coverage
- Idiomatic code style

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| HumanEval | Pass@1 | ≥ 70% | GPT-4: ~85% |
| MBPP | Pass@1 | ≥ 65% | GPT-4: ~80% |
| LiveCodeBench | Pass@1 | ≥ 50% | GPT-4: ~60% |

### Acceptance Criteria
- HumanEval Pass@1 ≥ 70%
- MBPP Pass@1 ≥ 65%
- Runtime exception rate < 5%
- Syntax hallucination rate < 2%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Off-by-one | Boundary condition errors | `range(n)` vs `range(n+1)` |
| Missing base case | Recursion without termination | Infinite recursion |
| Null handling | Not checking for None/null | NoneType errors |
| Timeout | Inefficient algorithms | O(n²) when O(n) needed |

### Synthetic Data Priorities
- Function implementation from specs
- Algorithm implementation
- Edge case coverage examples
- Error handling patterns

---

## Bucket 20: Code Generation (Tier 2: Java, C++, TypeScript, Go)

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-GEN-T2` |
| **Phase** | `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Code generation for statically-typed and systems programming languages.

### Sub-skills
- Type annotation and inference
- Memory management (C++)
- Generics and templates
- Interface implementation
- Concurrency patterns
- Build system awareness

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MultiPL-E (Java) | Pass@1 | ≥ 60% | GPT-4: ~70% |
| MultiPL-E (C++) | Pass@1 | ≥ 55% | GPT-4: ~65% |
| MultiPL-E (Go) | Pass@1 | ≥ 60% | GPT-4: ~70% |
| MultiPL-E (TypeScript) | Pass@1 | ≥ 65% | GPT-4: ~75% |

### Acceptance Criteria
- Pass@1 within 15% of Python performance
- Type correctness ≥ 95%
- Memory safety (C++) ≥ 90%
- Compilation success ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Type mismatch | Incompatible types | String where int expected |
| Memory leak | Missing deallocation (C++) | new without delete |
| Null pointer | Uninitialized references | NullPointerException |
| Generic constraints | Wrong type bounds | Missing interface implementation |

### Synthetic Data Priorities
- Type-annotated implementations
- Memory management patterns (C++)
- Generic/template usage
- Interface implementations

---

## Bucket 21: Code Generation (Tier 3: Rust, SQL, Bash, Others)

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-GEN-T3` |
| **Phase** | `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Specialized languages with unique paradigms requiring domain-specific knowledge.

### Sub-skills
- Ownership and borrowing (Rust)
- Query optimization (SQL)
- Shell scripting (Bash)
- Domain-specific idioms
- Safety patterns

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MultiPL-E (Rust) | Pass@1 | ≥ 45% | GPT-4: ~55% |
| Spider (SQL) | Execution accuracy | ≥ 75% | GPT-4: ~80% |
| Internal (Bash) | Functional | ≥ 80% | — |

### Acceptance Criteria
- Rust: Compilation success ≥ 70%
- SQL: Query correctness ≥ 75%
- Bash: Functional scripts ≥ 80%
- No security vulnerabilities (SQL injection, etc.)

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Borrow checker | Ownership violations (Rust) | Multiple mutable borrows |
| SQL injection | Unparameterized queries | String concatenation in queries |
| Unquoted variables | Bash word splitting | `$var` vs `"$var"` |
| Missing error handling | No error checks in scripts | Commands assumed to succeed |

### Synthetic Data Priorities
- Rust ownership patterns
- Safe SQL query construction
- Bash scripting best practices
- Error handling in scripts

---

## Bucket 22: Code Debugging & Error Correction

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-DBG` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Identifying bugs in existing code, understanding error messages, and generating correct fixes.

### Sub-skills
- Bug localization
- Error message interpretation
- Fix generation
- Regression prevention
- Root cause analysis
- Minimal fix identification

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| DebugBench | Localization accuracy | ≥ 80% | GPT-4: ~75% |
| SWE-bench (subset) | Fix rate | ≥ 25% | GPT-4: ~30% |
| Internal bug probes | Fix success | ≥ 70% | — |

### Acceptance Criteria
- Bug localization accuracy ≥ 80%
- Fix success rate ≥ 70%
- No regression introduction ≥ 90%
- Root cause identification ≥ 75%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Symptom fixing | Addressing symptom not cause | Adding try-catch instead of fixing logic |
| New bugs | Fix introduces new issues | Breaking other functionality |
| Over-engineering | Unnecessarily complex fixes | Rewriting when one-line change suffices |
| Related bug blindness | Missing connected issues | Not seeing pattern of similar bugs |

### Synthetic Data Priorities
- Bug localization exercises
- Minimal fix identification
- Error message interpretation
- Regression test generation

---

## Bucket 23: Code Optimization & Refactoring

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-OPT` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B4-B5 |
| **Critical** | No |

### Description
Improving code efficiency, readability, and maintainability while preserving correctness.

### Sub-skills
- Time complexity improvement
- Space complexity improvement
- Code deduplication
- Design pattern application
- Readability enhancement
- API modernization

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| PIE benchmark | Improvement rate | ≥ 70% | GPT-4: ~75% |
| Internal optimization | Correctness preserved | 100% | — |
| — | Measurable improvement | ≥ 80% | — |

### Acceptance Criteria
- Optimization preserves correctness 100%
- Measurable improvement in ≥ 80% of cases
- No functionality regression
- Readability maintained or improved

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Correctness break | Optimization changes behavior | Edge case now fails |
| Premature optimization | Optimizing non-bottlenecks | Complex code for rare path |
| Readability loss | Clever but unreadable | One-liners nobody understands |
| Over-engineering | Adding unnecessary abstraction | Patterns where simple code works |

### Synthetic Data Priorities
- Before/after optimization pairs
- Complexity analysis exercises
- Refactoring patterns
- Correctness-preserving transformations

---

## Bucket 24: Test Generation & Verification

| Attribute | Value |
|-----------|-------|
| **ID** | `CODE-TEST` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Generating test cases, writing assertions, and creating verification logic for code.

### Sub-skills
- Unit test generation
- Edge case identification
- Assertion writing
- Test coverage awareness
- Property-based testing concepts
- Mock and stub creation

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| EvalPlus | Bug detection | ≥ 80% | GPT-4: ~70% |
| Internal test gen | Edge coverage | ≥ 75% | — |
| — | False positive rate | < 5% | — |

### Acceptance Criteria
- Generated tests catch known bugs ≥ 80%
- Edge case coverage ≥ 75%
- False positive rate < 5%
- Test independence maintained

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Trivial tests | Only testing happy path | No edge cases |
| Missing boundaries | Not testing limits | Off-by-one not caught |
| Wrong expected values | Assertions with wrong answers | Test passes on buggy code |
| Flaky tests | Non-deterministic failures | Random/timing dependencies |

### Synthetic Data Priorities
- Test case generation from specs
- Edge case enumeration
- Assertion writing exercises
- Bug-detecting test examples

---

# Plane 4: Language-Specific & Cross-Lingual

Language buckets handle Hindi-specific capabilities and cross-lingual transfer.

---

## Bucket 25: Hindi Comprehension (Devanagari)

| Attribute | Value |
|-----------|-------|
| **ID** | `LANG-HI-COMP` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B1-B4 |
| **Critical** | ✓ Yes |

### Description
Understanding Hindi text in Devanagari script, including vocabulary, idioms, and register.

### Sub-skills
- Devanagari script recognition and parsing
- Vocabulary comprehension
- Idiomatic expression understanding
- Register awareness (formal/informal/literary)
- Regional variation handling
- Cultural context understanding

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MMLU-Hi (translated) | Accuracy | ≥ 60% | GPT-4: ~65% |
| IndicNLP Suite | Various | ≥ 75% | — |
| Internal Hindi probes | Comprehension | ≥ 85% | — |
| — | Delta vs English | ≤ 5% | GPT-4: ~5-8% |

### Acceptance Criteria
- Hindi comprehension within 5% of English
- Script parsing accuracy 100%
- Idiom understanding ≥ 80%
- Register identification ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Character confusion | Similar Devanagari letters | ब/व, घ/ध confusion |
| Formal register miss | Not recognizing formal Hindi | Missing शुद्ध हिंदी context |
| Idiom literal | Taking मुहावरे literally | "हाथ पैर मारना" as physical action |
| Regional blindness | Missing dialect markers | Not recognizing regional variants |

### Synthetic Data Priorities
- Devanagari script exercises
- Idiomatic expression pairs
- Register identification
- Reading comprehension passages

---

## Bucket 26: Hindi Generation (Devanagari)

| Attribute | Value |
|-----------|-------|
| **ID** | `LANG-HI-GEN` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B1-B4 |
| **Critical** | No |

### Description
Producing fluent, grammatically correct Hindi text in Devanagari script.

### Sub-skills
- Grammatically correct generation
- Natural phrasing and flow
- Appropriate register selection
- Proper matra (vowel mark) usage
- Sentence variety
- Avoiding English intrusion

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| Human evaluation | Fluency | ≥ 4.0/5.0 | GPT-4: ~4.2 |
| Internal probes | Grammar accuracy | ≥ 95% | — |
| — | Code-point validity | 100% | — |

### Acceptance Criteria
- Grammar accuracy ≥ 95%
- Fluency score ≥ 4.0/5.0 (human eval)
- Unicode code-point validity 100%
- Minimal unnecessary English

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Gender agreement | Verb-noun gender mismatch | As in FND-LEX-HI |
| Unnatural ordering | English word order in Hindi | SVO instead of SOV |
| English overuse | Unnecessary anglicisms | Using English words with Hindi equivalents |
| Matra errors | Incorrect vowel diacritics | Wrong vowel marks |

### Synthetic Data Priorities
- Fluent Hindi passages
- Grammar-focused exercises
- Register-appropriate generation
- Script correctness validation

---

## Bucket 27: English-Hindi Translation

| Attribute | Value |
|-----------|-------|
| **ID** | `LANG-TRANS` |
| **Phase** | `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | No |

### Description
Bidirectional translation between English and Hindi with semantic preservation.

### Sub-skills
- Semantic preservation
- Idiomatic adaptation (not literal translation)
- Technical term handling
- Cultural context preservation
- Formality matching
- Transliteration when appropriate

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| FLORES-200 (En→Hi) | BLEU | ≥ 35 | Google: ~38 |
| FLORES-200 (Hi→En) | BLEU | ≥ 40 | Google: ~42 |
| WMT Hindi | BLEU | ≥ 30 | — |
| Human evaluation | Adequacy | ≥ 4.0/5.0 | — |

### Acceptance Criteria
- BLEU ≥ 35 (En→Hi)
- BLEU ≥ 40 (Hi→En)
- Semantic preservation ≥ 90%
- No critical meaning errors

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Literal idiom | Translating idioms word-by-word | "Kick the bucket" → "बाल्टी को लात मारो" |
| Formality loss | Not matching register | Formal English → informal Hindi |
| Technical error | Wrong domain terms | Medical/legal terminology mistakes |
| Ambiguity creation | Clear source → ambiguous target | Losing disambiguation |

### Synthetic Data Priorities
- Parallel sentence pairs
- Idiomatic adaptation examples
- Technical domain translations
- Register-matched pairs

---

## Bucket 28: Code-Mixed Language (Hinglish)

| Attribute | Value |
|-----------|-------|
| **ID** | `LANG-MIX` |
| **Phase** | `[PRE]` `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | No |

### Description
Understanding and generating Hindi-English code-mixed text (Hinglish) common in Indian digital communication.

### Sub-skills
- Script-mixed parsing (Devanagari + Latin)
- Semantic coherence in mixed utterances
- Natural switch point identification
- Romanized Hindi understanding
- Transliteration handling
- Social media conventions

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| GLUECoS | Accuracy | ≥ 70% | Limited baselines |
| Hinglish sentiment | F1 | ≥ 75% | — |
| Internal code-mix probes | Semantic drift | < 3% | — |

### Acceptance Criteria
- Code-mix semantic drift < 3%
- Natural mixing patterns ≥ 80%
- Romanized Hindi accuracy ≥ 90%
- Script mixing handled seamlessly

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Unnatural switching | Breaking at wrong points | Switching mid-word inappropriately |
| Script confusion | Mixing conventions wrongly | Devanagari in English context |
| Meaning loss | Semantic drift in mixing | Different meaning than intended |
| Romanization errors | Inconsistent transliteration | Multiple spellings creating confusion |

### Synthetic Data Priorities
- Natural code-mix conversations
- Romanized Hindi normalization
- Script-mixed processing
- Social media style text

---

## Bucket 29: Hindi Logical Reasoning

| Attribute | Value |
|-----------|-------|
| **ID** | `LANG-HI-LOG` |
| **Phase** | `[SFT]` |
| **Bands** | B3-B5 |
| **Critical** | No |

### Description
Logical and deductive reasoning tasks presented entirely in Hindi.

### Sub-skills
- Syllogistic reasoning in Hindi
- Causal reasoning in Hindi
- Analogical reasoning in Hindi
- Logical connective understanding (इसलिए, क्योंकि, अगर...तो, हालांकि)
- Quantifier handling (सभी, कुछ, कोई नहीं)

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| BigBench-Hi (translated) | Accuracy | ≥ 70% | Limited baselines |
| Internal Hindi logic | Delta vs English | ≤ 8% | — |
| — | Connective understanding | ≥ 90% | — |

### Acceptance Criteria
- Hindi logic accuracy within 8% of English
- Logical connective understanding ≥ 90%
- Syllogistic reasoning ≥ 80%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Connective misread | Wrong interpretation of logical words | "हालांकि" (although) as "इसलिए" (therefore) |
| Cultural interference | Cultural assumptions affecting logic | Context-dependent reasoning errors |
| Quantifier confusion | Wrong scope of सभी/कुछ | Universal vs existential errors |

### Synthetic Data Priorities
- Hindi logic problems
- Connective-focused exercises
- Syllogisms in Hindi
- Reasoning chains in Hindi

---

# Plane 5: Alignment & Safety

Alignment buckets ensure the model follows instructions, avoids harm, and provides helpful responses.

---

## Bucket 30: Instruction Following

| Attribute | Value |
|-----------|-------|
| **ID** | `ALN-INST` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B2-B5 |
| **Critical** | ✓ Yes |

### Description
Accurately following user instructions, respecting constraints, and handling multi-part requests.

### Sub-skills
- Constraint adherence (length, format, style)
- Multi-instruction handling
- Implicit instruction inference
- Instruction prioritization
- Negative instruction handling ("don't", "avoid")
- Partial compliance recognition

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| IFEval (strict) | Accuracy | ≥ 70% | GPT-4: ~80% |
| IFEval (loose) | Accuracy | ≥ 85% | GPT-4: ~90% |
| MT-Bench (instruction) | Score | ≥ 8.0 | GPT-4: ~9.0 |

### Acceptance Criteria
- IFEval strict accuracy ≥ 70%
- Instruction adherence ≥ 95%
- Multi-constraint satisfaction ≥ 85%
- Negative instruction compliance ≥ 90%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Secondary constraint ignore | Missing non-primary requirements | Following format but wrong length |
| Partial following | Incomplete instruction execution | 2 of 3 requirements met |
| Over-interpretation | Adding unrequested elements | Embellishing simple requests |
| Negative miss | Ignoring "don't" instructions | Including forbidden content |

### Synthetic Data Priorities
- Multi-constraint instruction pairs
- Negative instruction examples
- Complex instruction decomposition
- Edge case instructions

---

## Bucket 31: Structured Output Generation

| Attribute | Value |
|-----------|-------|
| **ID** | `ALN-STRUCT` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B3-B5 |
| **Critical** | ✓ Yes |

### Description
Generating valid structured formats including JSON, XML, YAML, Markdown, and custom schemas.

### Sub-skills
- JSON validity and schema compliance
- XML well-formedness
- YAML correctness
- Markdown formatting
- Table generation
- API response formatting
- Custom schema adherence

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| IFEval (format) | JSON validity | ≥ 98% | Claude 3: ~98% |
| Internal schema tests | Schema compliance | ≥ 95% | — |
| — | Format consistency | ≥ 98% | — |

### Acceptance Criteria
- JSON validity ≥ 98%
- Schema compliance ≥ 95%
- Format consistency ≥ 98%
- No structural errors

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Trailing comma | JSON syntax error | `{"a": 1,}` |
| Unescaped characters | Special chars breaking format | Unescaped quotes in strings |
| Schema field omission | Missing required fields | Incomplete object |
| Inconsistent formatting | Mixed conventions | Different date formats |

### Synthetic Data Priorities
- JSON generation with schemas
- Complex nested structures
- Schema compliance exercises
- Format conversion tasks

---

## Bucket 32: Hallucination Resistance

| Attribute | Value |
|-----------|-------|
| **ID** | `ALN-HALL` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B2-B5 |
| **Critical** | ✓ Yes |

### Description
Avoiding fabricated information, expressing appropriate uncertainty, and maintaining calibrated confidence.

### Sub-skills
- Factual grounding
- Source attribution
- Uncertainty expression
- "I don't know" calibration
- Confidence calibration
- Abstention on unknowns

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| TruthfulQA | Accuracy | ≥ 65% | Claude 3: ~65% |
| HaluEval | Detection rate | ≥ 80% | — |
| Internal hallucination | Rate | < 10% | GPT-4: ~12% |
| — | Abstention precision | ≥ 85% | — |

### Acceptance Criteria
- TruthfulQA accuracy ≥ 65%
- Hallucination rate < 10%
- Abstention precision ≥ 85%
- Calibration error < 15%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Confident fabrication | High confidence on false claims | Made-up statistics |
| Citation invention | Fabricated references | Non-existent papers |
| Statistic hallucination | Made-up numbers | "Studies show 73%..." |
| Entity conflation | Mixing up similar entities | Wrong attributions |

### Synthetic Data Priorities
- Uncertainty expression examples
- Abstention on unknowns
- Factual verification pairs
- Confidence calibration

---

## Bucket 33: Safety & Harm Avoidance

| Attribute | Value |
|-----------|-------|
| **ID** | `ALN-SAFE` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B0-B5 |
| **Critical** | ✓ Yes |

### Description
Refusing harmful requests, avoiding dangerous content generation, and resisting manipulation attempts.

### Sub-skills
- Harmful request detection
- Appropriate refusal generation
- Jailbreak resistance
- Indirect harm recognition
- Dual-use content handling
- Adversarial robustness

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| Red-team suite | Harmful generation | < 1% | Claude 3: <1% |
| Jailbreak tests | Resistance | ≥ 95% | — |
| Internal safety | False refusal | < 5% | — |
| — | Missed refusal | < 3% | — |

### Acceptance Criteria
- Harmful content generation rate < 1%
- False refusal rate < 5%
- Missed refusal rate < 3%
- Jailbreak resistance ≥ 95%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Jailbreak success | Manipulation bypasses safety | Role-play exploits |
| Over-refusal | Blocking benign requests | Refusing cooking chemistry |
| Indirect harm | Helping via obfuscation | Coded language assistance |
| Context manipulation | Tricked by false context | "For my novel..." exploits |

### Synthetic Data Priorities
- Harmful request refusals
- Jailbreak resistance examples
- Benign-but-suspicious queries
- Dual-use content handling

---

## Bucket 34: Helpfulness & Engagement

| Attribute | Value |
|-----------|-------|
| **ID** | `ALN-HELP` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B1-B5 |
| **Critical** | No |

### Description
Providing genuinely useful, engaging, and appropriately detailed responses.

### Sub-skills
- Query understanding
- Comprehensive answers
- Appropriate detail level
- Proactive clarification
- Conversational naturalness
- Task completion focus

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| MT-Bench | Score | ≥ 8.0/10 | GPT-4: ~9.0 |
| AlpacaEval | Win rate | ≥ 50% | GPT-4: ~95% |
| Human evaluation | Satisfaction | ≥ 4.0/5.0 | — |

### Acceptance Criteria
- MT-Bench score ≥ 8.0/10
- AlpacaEval win rate ≥ 50%
- User satisfaction ≥ 4.0/5.0
- Task completion rate ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Over-verbosity | Too much unnecessary detail | Simple question → essay |
| Question miss | Not addressing actual query | Answering adjacent topic |
| Unhelpful hedging | Excessive caveats | "It depends..." without guidance |
| Robotic tone | Unnatural, stilted responses | Template-like answers |

### Synthetic Data Priorities
- High-quality response examples
- Appropriate length calibration
- Natural conversation flows
- Task completion demonstrations

---

# Plane 6: Production & Robustness

Production buckets cover real-world deployment requirements and robustness.

---

## Bucket 35: Input Robustness

| Attribute | Value |
|-----------|-------|
| **ID** | `PRD-ROB` |
| **Phase** | `[ALL]` |
| **Bands** | B1-B5 |
| **Critical** | No |

### Description
Handling malformed, adversarial, or unusual inputs gracefully without performance collapse.

### Sub-skills
- Typo tolerance
- Grammar error handling
- Adversarial input detection
- Unicode edge cases
- Empty/null input handling
- Prompt injection resistance

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| CheckList | Robustness | ≥ 85% | GPT-4: ~90% |
| Adversarial NLI | Accuracy | ≥ 70% | GPT-4: ~75% |
| Internal robustness | Degradation | < 10% | — |

### Acceptance Criteria
- Performance degradation < 10% on noisy input
- Adversarial accuracy ≥ 70%
- No crashes on edge cases
- Graceful handling of malformed input

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Typo catastrophe | Complete failure on typos | Minor spelling breaks understanding |
| Adversarial exploit | Manipulation via input | Prompt injection success |
| Unicode crash | Special characters cause issues | Emoji/special char handling |
| Empty input failure | No graceful null handling | Crashes on empty prompt |

### Synthetic Data Priorities
- Typo-variant examples
- Adversarial input handling
- Unicode edge cases
- Graceful error responses

---

## Bucket 36: Summarization

| Attribute | Value |
|-----------|-------|
| **ID** | `PRD-SUM` |
| **Phase** | `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | No |

### Description
Condensing longer texts while preserving key information and factual accuracy.

### Sub-skills
- Key point extraction
- Length control
- Factual preservation
- Abstractive vs extractive balance
- Multi-document summarization
- Query-focused summarization

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| CNN/DailyMail | ROUGE-L | ≥ 35 | GPT-4: ~40 |
| XSum | ROUGE-L | ≥ 30 | GPT-4: ~35 |
| Human evaluation | Factual consistency | ≥ 90% | — |

### Acceptance Criteria
- ROUGE-L ≥ 35
- Factual consistency ≥ 90%
- Length adherence ≥ 95%
- Key point coverage ≥ 85%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Hallucinated details | Adding non-source information | Inventing quotes |
| Key point omission | Missing critical information | Leaving out main finding |
| Extractive over-reliance | Just copying sentences | No abstraction |
| Length violation | Wrong output length | Too long or too short |

### Synthetic Data Priorities
- Document-summary pairs
- Length-controlled summaries
- Factual consistency examples
- Multi-document synthesis

---

## Bucket 37: Information Extraction

| Attribute | Value |
|-----------|-------|
| **ID** | `PRD-IE` |
| **Phase** | `[SFT]` |
| **Bands** | B2-B4 |
| **Critical** | No |

### Description
Extracting structured information from unstructured text including entities, relations, and events.

### Sub-skills
- Named entity recognition (NER)
- Relation extraction
- Event extraction
- Slot filling
- Template population
- Attribute extraction

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| CoNLL NER | F1 | ≥ 90% | GPT-4: ~93% |
| TACRED | F1 | ≥ 75% | GPT-4: ~80% |
| Internal extraction | Slot filling accuracy | ≥ 80% | — |

### Acceptance Criteria
- NER F1 ≥ 90%
- Relation extraction F1 ≥ 75%
- Slot filling accuracy ≥ 80%
- Low false positive rate < 5%

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Boundary errors | Wrong entity span | "New" instead of "New York" |
| Relation direction | Reversed relation | A→B instead of B→A |
| Implicit entity miss | Not extracting implied entities | Missing referenced entities |
| Over-extraction | Extracting non-entities | Common nouns as entities |

### Synthetic Data Priorities
- NER annotation exercises
- Relation extraction pairs
- Slot filling templates
- Complex entity scenarios

---

## Bucket 38: Creative Writing

| Attribute | Value |
|-----------|-------|
| **ID** | `PRD-CREAT` |
| **Phase** | `[SFT]` `[RLHF]` |
| **Bands** | B2-B5 |
| **Critical** | No |

### Description
Generating creative, engaging, original content across various formats and styles.

### Sub-skills
- Narrative construction
- Style adaptation
- Tone control
- Creativity and originality
- Genre awareness
- Character and dialogue

### Primary Benchmarks
| Benchmark | Metric | Target | SOTA |
|-----------|--------|--------|------|
| Human evaluation | Creativity | ≥ 4.0/5.0 | GPT-4: ~4.2 |
| — | Engagement | ≥ 4.0/5.0 | — |
| WritingPrompts | Quality | ≥ 3.5/5.0 | — |
| Style adherence | Accuracy | ≥ 85% | — |

### Acceptance Criteria
- Creativity score ≥ 4.0/5.0 (human eval)
- Style adherence ≥ 85%
- Engagement rating ≥ 4.0/5.0
- No repetitive patterns

### Common Failure Modes
| Failure | Description | Example |
|---------|-------------|---------|
| Repetitive patterns | Same structures/phrases | Predictable openings |
| Originality lack | Generic, template responses | Cliché-heavy writing |
| Tone inconsistency | Shifting voice mid-piece | Formal→casual drift |
| Plot holes | Narrative inconsistencies | Contradictory story elements |

### Synthetic Data Priorities
- Diverse creative examples
- Style variation demonstrations
- Genre-specific writing
- Originality-focused pieces

---

## Appendix A: Acronyms & Abbreviations

| Acronym | Meaning |
|---------|---------|
| MoE | Mixture of Experts |
| CoT | Chain of Thought |
| SFT | Supervised Fine-Tuning |
| RLHF | Reinforcement Learning from Human Feedback |
| PRE | Pretraining |
| NER | Named Entity Recognition |
| BLEU | Bilingual Evaluation Understudy |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| QA | Question Answering |
| F1 | F1 Score (harmonic mean of precision and recall) |
| SOV | Subject-Object-Verb (Hindi word order) |
| SVO | Subject-Verb-Object (English word order) |

---

## Appendix B: Benchmark References

| Benchmark | Citation |
|-----------|----------|
| MMLU | Hendrycks et al., 2021 |
| GSM8K | Cobbe et al., 2021 |
| MATH | Hendrycks et al., 2021 |
| HumanEval | Chen et al., 2021 |
| MBPP | Austin et al., 2021 |
| IFEval | Zhou et al., 2023 |
| TruthfulQA | Lin et al., 2022 |
| HellaSwag | Zellers et al., 2019 |
| ARC | Clark et al., 2018 |
| BigBench | Srivastava et al., 2022 |
| LongBench | Bai et al., 2023 |
| FLORES-200 | Costa-jussà et al., 2022 |
| WinoGrande | Sakaguchi et al., 2020 |
| COPA | Roemmele et al., 2011 |
| HotpotQA | Yang et al., 2018 |
| Spider | Yu et al., 2018 |
| MT-Bench | Zheng et al., 2023 |

---

*Document Version: 2.0*  
*Last Updated: January 2025*  
*Team: Synthetic Data & Self-Distillation (Team 4)*
