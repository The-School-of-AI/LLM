"""
skills.py — Skill Buckets & Failure Mode Mapping (45 Curated Skills)

Maps: Benchmark → Skill → Failure Mode

Skill Categories:
- Foundation (FND): Language fundamentals, semantics, discourse
- Reasoning (RSN): Logic, math, causal, analogical
- Code (CODE): Syntax, generation, debugging, optimization
- Language (LANG): Hindi, translation, code-mixing
- Alignment (ALN): Instruction following, safety, hallucination
- Production (PRD): Summarization, extraction, robustness
- Indic (INDIC): Indian language benchmarks (IndicGLUE, IndicQA)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .bands import Band, Stage

# ================================================================
# SKILL CATEGORIES
# ================================================================


class SkillCategory(Enum):
    """High-level skill categories."""

    FOUNDATION = "foundation"
    REASONING = "reasoning"
    CODE = "code"
    LANGUAGE = "language"
    ALIGNMENT = "alignment"
    PRODUCTION = "production"
    INDIC = "indic"


# ================================================================
# FAILURE MODES
# ================================================================


@dataclass
class FailureMode:
    """A specific way a model can fail at a skill."""

    id: str
    description: str
    detection_signal: str  # How to detect this failure
    severity: Literal["critical", "high", "medium", "low"]


# Common failure modes (reusable across skills)
FAILURE_MODES = {
    # Reasoning failures
    "state_loss": FailureMode(
        id="state_loss",
        description="Loses track of state across reasoning steps",
        detection_signal="Contradicts earlier statements or forgets constraints",
        severity="critical",
    ),
    "step_skip": FailureMode(
        id="step_skip",
        description="Skips necessary intermediate steps",
        detection_signal="Jumps to conclusion without showing work",
        severity="high",
    ),
    "arithmetic_error": FailureMode(
        id="arithmetic_error",
        description="Makes basic arithmetic mistakes",
        detection_signal="Numerical answer is wrong despite correct approach",
        severity="high",
    ),
    "logic_violation": FailureMode(
        id="logic_violation",
        description="Violates logical rules (modus ponens, transitivity, etc)",
        detection_signal="Conclusion doesn't follow from premises",
        severity="critical",
    ),
    "contradiction_blind": FailureMode(
        id="contradiction_blind",
        description="Fails to detect contradictions in premises",
        detection_signal="Accepts contradictory statements as valid",
        severity="critical",
    ),
    "causal_reversal": FailureMode(
        id="causal_reversal",
        description="Confuses cause and effect",
        detection_signal="States effect as cause or vice versa",
        severity="high",
    ),
    "multi_hop_failure": FailureMode(
        id="multi_hop_failure",
        description="Fails to connect multiple pieces of evidence",
        detection_signal="Answers based on single hop when multiple required",
        severity="high",
    ),
    # Code failures
    "var_hallucination": FailureMode(
        id="var_hallucination",
        description="Hallucinates variable names not in scope",
        detection_signal="Uses undefined variables",
        severity="critical",
    ),
    "syntax_error": FailureMode(
        id="syntax_error",
        description="Generates syntactically invalid code",
        detection_signal="Code doesn't parse",
        severity="high",
    ),
    "type_mismatch": FailureMode(
        id="type_mismatch",
        description="Mismatches types in operations",
        detection_signal="Type error at runtime or static analysis",
        severity="high",
    ),
    "off_by_one": FailureMode(
        id="off_by_one",
        description="Off-by-one errors in loops/indices",
        detection_signal="Boundary conditions fail",
        severity="medium",
    ),
    "incomplete_logic": FailureMode(
        id="incomplete_logic",
        description="Missing edge cases or branches",
        detection_signal="Fails on edge inputs",
        severity="high",
    ),
    "inefficient_algorithm": FailureMode(
        id="inefficient_algorithm",
        description="Uses suboptimal time/space complexity",
        detection_signal="TLE or MLE on large inputs",
        severity="medium",
    ),
    "test_coverage_gap": FailureMode(
        id="test_coverage_gap",
        description="Generated tests miss important cases",
        detection_signal="Mutation testing finds gaps",
        severity="medium",
    ),
    # Language failures
    "fluency_break": FailureMode(
        id="fluency_break",
        description="Breaks grammatical fluency",
        detection_signal="Ungrammatical sentences",
        severity="medium",
    ),
    "coherence_loss": FailureMode(
        id="coherence_loss",
        description="Loses coherence across sentences",
        detection_signal="Sentences don't connect logically",
        severity="high",
    ),
    "repetition": FailureMode(
        id="repetition",
        description="Excessive repetition of phrases",
        detection_signal="Same phrase repeated 3+ times",
        severity="medium",
    ),
    "translation_error": FailureMode(
        id="translation_error",
        description="Incorrect translation preserving meaning",
        detection_signal="Semantic drift in translation",
        severity="high",
    ),
    "code_switch_error": FailureMode(
        id="code_switch_error",
        description="Incorrect code-switching in mixed language",
        detection_signal="Unnatural language mixing",
        severity="medium",
    ),
    # Knowledge failures
    "factual_error": FailureMode(
        id="factual_error",
        description="States incorrect facts",
        detection_signal="Verifiably false claims",
        severity="high",
    ),
    "temporal_confusion": FailureMode(
        id="temporal_confusion",
        description="Confuses temporal ordering of events",
        detection_signal="Gets before/after relationships wrong",
        severity="medium",
    ),
    "context_loss": FailureMode(
        id="context_loss",
        description="Loses context in long documents",
        detection_signal="Answers contradict earlier context",
        severity="high",
    ),
    # Alignment failures
    "instruction_violation": FailureMode(
        id="instruction_violation",
        description="Fails to follow explicit instructions",
        detection_signal="Output doesn't match instruction constraints",
        severity="critical",
    ),
    "format_violation": FailureMode(
        id="format_violation",
        description="Fails to produce required output format",
        detection_signal="JSON/XML/structured output malformed",
        severity="high",
    ),
    "hallucination": FailureMode(
        id="hallucination",
        description="Generates false information confidently",
        detection_signal="Fabricated facts, citations, or data",
        severity="critical",
    ),
    "safety_violation": FailureMode(
        id="safety_violation",
        description="Generates harmful or unsafe content",
        detection_signal="Content violates safety guidelines",
        severity="critical",
    ),
    "unhelpful": FailureMode(
        id="unhelpful",
        description="Response doesn't address user need",
        detection_signal="User would need to re-ask",
        severity="medium",
    ),
    # Production failures
    "summarization_loss": FailureMode(
        id="summarization_loss",
        description="Loses key information in summary",
        detection_signal="Important facts omitted",
        severity="high",
    ),
    "extraction_error": FailureMode(
        id="extraction_error",
        description="Extracts wrong entities or relations",
        detection_signal="NER/RE output incorrect",
        severity="high",
    ),
    "robustness_failure": FailureMode(
        id="robustness_failure",
        description="Fails on perturbed or noisy input",
        detection_signal="Performance drops on typos/paraphrases",
        severity="medium",
    ),
    # Indic-specific failures
    "indic_script_error": FailureMode(
        id="indic_script_error",
        description="Errors in Devanagari or other Indic scripts",
        detection_signal="Incorrect character rendering or ordering",
        severity="high",
    ),
    "indic_morphology_error": FailureMode(
        id="indic_morphology_error",
        description="Incorrect morphological forms in Indic languages",
        detection_signal="Wrong verb conjugation, gender agreement",
        severity="medium",
    ),
}


# ================================================================
# SKILL BUCKETS
# ================================================================


@dataclass
class SkillBucket:
    """A skill bucket that synthetic data targets."""

    id: str
    name: str
    category: SkillCategory
    description: str

    # What benchmarks test this skill
    related_benchmarks: list[str]

    # Expected failure modes for this skill
    failure_modes: list[str]  # IDs from FAILURE_MODES

    # Band alignment
    min_band: Band
    max_band: Band
    primary_band: Band  # Most relevant band

    # Stage alignment
    stages: list[Stage]

    # Language
    languages: list[str] = field(default_factory=lambda: ["en"])

    # Synthetic data requirements
    target_tokens: int = 5_000_000  # 5M tokens default
    priority: Literal["critical", "high", "medium", "low"] = "high"


# ================================================================
# 45 CURATED SKILL BUCKETS
# ================================================================

SKILL_BUCKETS: dict[str, SkillBucket] = {
    # ═══════════════════════════════════════════════════════════════
    # FOUNDATION SKILLS (6)
    # ═══════════════════════════════════════════════════════════════
    "FND-LEX-EN": SkillBucket(
        id="FND-LEX-EN",
        name="English Lexical & Syntactic",
        category=SkillCategory.FOUNDATION,
        description="English grammar, syntax, high-frequency constructions",
        related_benchmarks=["HellaSwag", "BLiMP", "CoLA"],
        failure_modes=["fluency_break", "syntax_error"],
        min_band=Band.B0,
        max_band=Band.B2,
        primary_band=Band.B1,
        stages=[Stage.PRE],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "FND-LEX-HI": SkillBucket(
        id="FND-LEX-HI",
        name="Hindi Lexical & Syntactic",
        category=SkillCategory.FOUNDATION,
        description="Hindi grammar, syntax, Devanagari script fluency",
        related_benchmarks=["HellaSwag-Hi", "IndicGLUE"],
        failure_modes=["fluency_break", "indic_script_error", "indic_morphology_error"],
        min_band=Band.B0,
        max_band=Band.B2,
        primary_band=Band.B1,
        stages=[Stage.PRE],
        languages=["hi"],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "FND-SEM": SkillBucket(
        id="FND-SEM",
        name="Semantic Understanding",
        category=SkillCategory.FOUNDATION,
        description="Word meaning, sentence semantics, entailment",
        related_benchmarks=["WinoGrande", "COPA", "WSC"],
        failure_modes=["logic_violation", "coherence_loss"],
        min_band=Band.B1,
        max_band=Band.B3,
        primary_band=Band.B2,
        stages=[Stage.PRE],
        target_tokens=5_000_000,
        priority="high",
    ),
    "FND-DIS": SkillBucket(
        id="FND-DIS",
        name="Discourse & Coherence",
        category=SkillCategory.FOUNDATION,
        description="Cross-sentence coherence, discourse structure",
        related_benchmarks=["WSC", "StoryCloze"],
        failure_modes=["coherence_loss", "context_loss"],
        min_band=Band.B2,
        max_band=Band.B3,
        primary_band=Band.B2,
        stages=[Stage.PRE],
        target_tokens=5_000_000,
        priority="high",
    ),
    "FND-LCX": SkillBucket(
        id="FND-LCX",
        name="Long-Context Retention",
        category=SkillCategory.FOUNDATION,
        description="Maintaining context over long documents (8K+ tokens)",
        related_benchmarks=["LongBench", "SCROLLS", "QuALITY"],
        failure_modes=["context_loss", "state_loss"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "FND-FACT": SkillBucket(
        id="FND-FACT",
        name="Factual Knowledge Encoding",
        category=SkillCategory.FOUNDATION,
        description="World knowledge, factual recall",
        related_benchmarks=["TriviaQA", "NaturalQuestions", "WebQuestions"],
        failure_modes=["factual_error", "temporal_confusion"],
        min_band=Band.B1,
        max_band=Band.B3,
        primary_band=Band.B2,
        stages=[Stage.PRE],
        target_tokens=8_000_000,
        priority="high",
    ),
    # ═══════════════════════════════════════════════════════════════
    # REASONING SKILLS (11)
    # ═══════════════════════════════════════════════════════════════
    "RSN-LOG": SkillBucket(
        id="RSN-LOG",
        name="Logical Deduction",
        category=SkillCategory.REASONING,
        description="Syllogisms, modus ponens, logical chains, proof steps",
        related_benchmarks=["BigBench-Logic", "LogiQA", "ReClor", "FOLIO"],
        failure_modes=["logic_violation", "contradiction_blind", "state_loss"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "RSN-CAUS": SkillBucket(
        id="RSN-CAUS",
        name="Causal Reasoning",
        category=SkillCategory.REASONING,
        description="Cause-effect chains, counterfactuals, temporal ordering",
        related_benchmarks=["COPA", "e-CARE", "WINOGRANDE"],
        failure_modes=["causal_reversal", "temporal_confusion", "logic_violation"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=5_000_000,
        priority="high",
    ),
    "RSN-ARITH": SkillBucket(
        id="RSN-ARITH",
        name="Arithmetic & Numerical",
        category=SkillCategory.REASONING,
        description="Multi-step arithmetic, numerical reasoning",
        related_benchmarks=["GSM8K", "SVAMP", "MAWPS"],
        failure_modes=["arithmetic_error", "step_skip", "state_loss"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "RSN-ALG": SkillBucket(
        id="RSN-ALG",
        name="Algebraic & Symbolic",
        category=SkillCategory.REASONING,
        description="Variable manipulation, equation solving, symbolic reasoning",
        related_benchmarks=["MATH-Algebra", "MMLU-Math"],
        failure_modes=["state_loss", "step_skip", "logic_violation"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=8_000_000,
        priority="high",
    ),
    "RSN-WPT": SkillBucket(
        id="RSN-WPT",
        name="Word Problem Translation",
        category=SkillCategory.REASONING,
        description="Translating word problems to mathematical formulation",
        related_benchmarks=["GSM8K", "AQUA-RAT", "MathQA"],
        failure_modes=["step_skip", "state_loss", "arithmetic_error"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "RSN-ADVMATH": SkillBucket(
        id="RSN-ADVMATH",
        name="Advanced Mathematical",
        category=SkillCategory.REASONING,
        description="Competition math, proofs, advanced problem solving",
        related_benchmarks=["MATH", "AMC", "AIME"],
        failure_modes=["logic_violation", "step_skip", "state_loss"],
        min_band=Band.B4,
        max_band=Band.B5,
        primary_band=Band.B5,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    "RSN-MATH-HI": SkillBucket(
        id="RSN-MATH-HI",
        name="Mathematical Reasoning (Hindi)",
        category=SkillCategory.REASONING,
        description="Math word problems in Hindi",
        related_benchmarks=["GSM8K-Hi", "IndicMath"],
        failure_modes=["arithmetic_error", "step_skip", "translation_error"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT],
        languages=["hi"],
        target_tokens=5_000_000,
        priority="critical",
    ),
    "RSN-CS": SkillBucket(
        id="RSN-CS",
        name="Commonsense Reasoning",
        category=SkillCategory.REASONING,
        description="Everyday world knowledge and physical reasoning",
        related_benchmarks=["HellaSwag", "PIQA", "ARC", "CommonsenseQA"],
        failure_modes=["factual_error", "logic_violation", "temporal_confusion"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=8_000_000,
        priority="high",
    ),
    "RSN-MH": SkillBucket(
        id="RSN-MH",
        name="Multi-Hop Reasoning",
        category=SkillCategory.REASONING,
        description="Connecting multiple pieces of evidence",
        related_benchmarks=["HotpotQA", "2WikiMultiHopQA", "MuSiQue"],
        failure_modes=["multi_hop_failure", "state_loss", "context_loss"],
        min_band=Band.B4,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    "RSN-CONTRADICTION": SkillBucket(
        id="RSN-CONTRADICTION",
        name="Contradiction Detection",
        category=SkillCategory.REASONING,
        description="Identifying logical contradictions in text",
        related_benchmarks=["ANLI", "SNLI", "MNLI"],
        failure_modes=["contradiction_blind", "logic_violation"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=5_000_000,
        priority="critical",
    ),
    "RSN-ANAL": SkillBucket(
        id="RSN-ANAL",
        name="Analogical Reasoning",
        category=SkillCategory.REASONING,
        description="A:B::C:D relationships, pattern matching",
        related_benchmarks=["BigBench-Analogy", "BATS", "SAT-Analogy"],
        failure_modes=["logic_violation", "step_skip"],
        min_band=Band.B3,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=3_000_000,
        priority="medium",
    ),
    # ═══════════════════════════════════════════════════════════════
    # CODE SKILLS (9)
    # ═══════════════════════════════════════════════════════════════
    "CODE-SYN": SkillBucket(
        id="CODE-SYN",
        name="Code Syntax & Structure",
        category=SkillCategory.CODE,
        description="Understanding code syntax, AST structure",
        related_benchmarks=["CodeXGLUE-Clone"],
        failure_modes=["syntax_error", "var_hallucination"],
        min_band=Band.B2,
        max_band=Band.B3,
        primary_band=Band.B2,
        stages=[Stage.PRE],
        target_tokens=5_000_000,
        priority="critical",
    ),
    "CODE-COMP": SkillBucket(
        id="CODE-COMP",
        name="Code Comprehension",
        category=SkillCategory.CODE,
        description="Understanding what code does",
        related_benchmarks=["MBPP", "CodeXGLUE"],
        failure_modes=["coherence_loss", "step_skip"],
        min_band=Band.B3,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "CODE-GEN-T1": SkillBucket(
        id="CODE-GEN-T1",
        name="Code Generation (Python, JS)",
        category=SkillCategory.CODE,
        description="Generate Python and JavaScript code",
        related_benchmarks=["HumanEval", "MBPP"],
        failure_modes=["var_hallucination", "syntax_error", "incomplete_logic"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "CODE-GEN-T2": SkillBucket(
        id="CODE-GEN-T2",
        name="Code Generation (Java, C++, TS, Go)",
        category=SkillCategory.CODE,
        description="Generate Java, C++, TypeScript, Go code",
        related_benchmarks=["MultiPL-E", "HumanEval-X"],
        failure_modes=["type_mismatch", "syntax_error", "var_hallucination"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT],
        target_tokens=8_000_000,
        priority="high",
    ),
    "CODE-GEN-T3": SkillBucket(
        id="CODE-GEN-T3",
        name="Code Generation (Rust, SQL, Bash)",
        category=SkillCategory.CODE,
        description="Generate Rust, SQL, Bash code",
        related_benchmarks=["Spider", "MultiPL-E"],
        failure_modes=["syntax_error", "type_mismatch", "incomplete_logic"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT],
        target_tokens=5_000_000,
        priority="high",
    ),
    "CODE-DBG": SkillBucket(
        id="CODE-DBG",
        name="Code Debugging & Error Correction",
        category=SkillCategory.CODE,
        description="Identify and fix bugs in code",
        related_benchmarks=["DebugBench", "BugNet"],
        failure_modes=["var_hallucination", "off_by_one", "type_mismatch"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "CODE-OPT": SkillBucket(
        id="CODE-OPT",
        name="Code Optimization & Refactoring",
        category=SkillCategory.CODE,
        description="Improve code efficiency and structure",
        related_benchmarks=["CodeContests"],
        failure_modes=["inefficient_algorithm", "incomplete_logic"],
        min_band=Band.B4,
        max_band=Band.B5,
        primary_band=Band.B5,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    "CODE-TEST": SkillBucket(
        id="CODE-TEST",
        name="Test Generation & Verification",
        category=SkillCategory.CODE,
        description="Generate unit tests, verify correctness",
        related_benchmarks=["EvalPlus", "TestGen"],
        failure_modes=["test_coverage_gap", "incomplete_logic"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    "CODE-ALGO": SkillBucket(
        id="CODE-ALGO",
        name="Algorithmic Implementation",
        category=SkillCategory.CODE,
        description="Implement algorithms from description",
        related_benchmarks=["CodeContests", "APPS"],
        failure_modes=["incomplete_logic", "off_by_one", "state_loss"],
        min_band=Band.B4,
        max_band=Band.B5,
        primary_band=Band.B5,
        stages=[Stage.PRE, Stage.SFT],
        target_tokens=8_000_000,
        priority="critical",
    ),
    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE SKILLS (6)
    # ═══════════════════════════════════════════════════════════════
    "LANG-HI-COMP": SkillBucket(
        id="LANG-HI-COMP",
        name="Hindi Comprehension",
        category=SkillCategory.LANGUAGE,
        description="Reading comprehension in Hindi",
        related_benchmarks=["MMLU-Hi", "IndicQA", "IndicGLUE"],
        failure_modes=["coherence_loss", "context_loss", "indic_morphology_error"],
        min_band=Band.B1,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        languages=["hi"],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "LANG-HI-GEN": SkillBucket(
        id="LANG-HI-GEN",
        name="Hindi Generation",
        category=SkillCategory.LANGUAGE,
        description="Fluent Hindi text generation",
        related_benchmarks=["IndicNLG"],
        failure_modes=["fluency_break", "indic_script_error", "indic_morphology_error"],
        min_band=Band.B1,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        languages=["hi"],
        target_tokens=8_000_000,
        priority="high",
    ),
    "LANG-TRANS": SkillBucket(
        id="LANG-TRANS",
        name="English-Hindi Translation",
        category=SkillCategory.LANGUAGE,
        description="Bidirectional translation EN<->HI",
        related_benchmarks=["FLORES-200", "WMT"],
        failure_modes=["translation_error", "fluency_break"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        languages=["en", "hi"],
        target_tokens=5_000_000,
        priority="high",
    ),
    "LANG-MIX": SkillBucket(
        id="LANG-MIX",
        name="Code-Mixed Language (Hinglish)",
        category=SkillCategory.LANGUAGE,
        description="Understanding and generating Hinglish",
        related_benchmarks=["GLUECoS", "LinCE"],
        failure_modes=["code_switch_error", "fluency_break"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT],
        languages=["hi-en"],
        target_tokens=5_000_000,
        priority="high",
    ),
    "LANG-HI-LOG": SkillBucket(
        id="LANG-HI-LOG",
        name="Hindi Logical Reasoning",
        category=SkillCategory.LANGUAGE,
        description="Logical reasoning in Hindi",
        related_benchmarks=["BigBench-Hi", "IndicLogic"],
        failure_modes=["logic_violation", "translation_error"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT],
        languages=["hi"],
        target_tokens=5_000_000,
        priority="high",
    ),
    "LANG-GRAMMAR": SkillBucket(
        id="LANG-GRAMMAR",
        name="Grammar & Syntax",
        category=SkillCategory.LANGUAGE,
        description="Correct grammatical constructions (B0-B1 coverage)",
        related_benchmarks=["CoLA", "BLiMP"],
        failure_modes=["fluency_break", "repetition"],
        min_band=Band.B0,
        max_band=Band.B2,
        primary_band=Band.B1,
        stages=[Stage.PRE],
        languages=["en", "hi"],
        target_tokens=5_000_000,
        priority="high",
    ),
    # ═══════════════════════════════════════════════════════════════
    # ALIGNMENT SKILLS (5)
    # ═══════════════════════════════════════════════════════════════
    "ALN-INST": SkillBucket(
        id="ALN-INST",
        name="Instruction Following",
        category=SkillCategory.ALIGNMENT,
        description="Following explicit instructions accurately",
        related_benchmarks=["IFEval", "InstructBench"],
        failure_modes=["instruction_violation", "step_skip"],
        min_band=Band.B2,
        max_band=Band.B5,
        primary_band=Band.B3,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "ALN-STRUCT": SkillBucket(
        id="ALN-STRUCT",
        name="Structured Output Generation",
        category=SkillCategory.ALIGNMENT,
        description="Generating JSON, XML, structured formats",
        related_benchmarks=["IFEval-Format", "JSONBench"],
        failure_modes=["format_violation", "syntax_error"],
        min_band=Band.B3,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="critical",
    ),
    "ALN-HALL": SkillBucket(
        id="ALN-HALL",
        name="Hallucination Resistance",
        category=SkillCategory.ALIGNMENT,
        description="Avoiding fabricated information",
        related_benchmarks=["TruthfulQA", "HaluEval"],
        failure_modes=["hallucination", "factual_error"],
        min_band=Band.B2,
        max_band=Band.B5,
        primary_band=Band.B4,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "ALN-SAFE": SkillBucket(
        id="ALN-SAFE",
        name="Safety & Harm Avoidance",
        category=SkillCategory.ALIGNMENT,
        description="Refusing harmful requests, safe responses",
        related_benchmarks=["Red-team", "SafetyBench"],
        failure_modes=["safety_violation"],
        min_band=Band.B0,
        max_band=Band.B5,
        primary_band=Band.B3,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="critical",
    ),
    "ALN-HELP": SkillBucket(
        id="ALN-HELP",
        name="Helpfulness & Engagement",
        category=SkillCategory.ALIGNMENT,
        description="Being genuinely helpful and engaging",
        related_benchmarks=["MT-Bench", "AlpacaEval"],
        failure_modes=["unhelpful", "repetition"],
        min_band=Band.B1,
        max_band=Band.B5,
        primary_band=Band.B3,
        stages=[Stage.SFT, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    # ═══════════════════════════════════════════════════════════════
    # PRODUCTION SKILLS (3)
    # ═══════════════════════════════════════════════════════════════
    "PRD-ROB": SkillBucket(
        id="PRD-ROB",
        name="Input Robustness",
        category=SkillCategory.PRODUCTION,
        description="Handling typos, paraphrases, noisy input",
        related_benchmarks=["CheckList", "TextFlint"],
        failure_modes=["robustness_failure"],
        min_band=Band.B1,
        max_band=Band.B5,
        primary_band=Band.B3,
        stages=[Stage.PRE, Stage.SFT, Stage.DPO, Stage.RLHF],
        target_tokens=5_000_000,
        priority="high",
    ),
    "PRD-SUM": SkillBucket(
        id="PRD-SUM",
        name="Summarization",
        category=SkillCategory.PRODUCTION,
        description="Summarizing documents accurately",
        related_benchmarks=["CNN/DM", "XSum", "SAMSum"],
        failure_modes=["summarization_loss", "hallucination"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        target_tokens=5_000_000,
        priority="high",
    ),
    "PRD-IE": SkillBucket(
        id="PRD-IE",
        name="Information Extraction",
        category=SkillCategory.PRODUCTION,
        description="Extracting entities, relations from text",
        related_benchmarks=["CoNLL", "TACRED", "DocRED"],
        failure_modes=["extraction_error", "format_violation"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        target_tokens=5_000_000,
        priority="high",
    ),
    # ═══════════════════════════════════════════════════════════════
    # INDIC SKILLS (5) - Based on IndicGLUE, IndicQA
    # ═══════════════════════════════════════════════════════════════
    "INDIC-QA": SkillBucket(
        id="INDIC-QA",
        name="Indic Question Answering",
        category=SkillCategory.INDIC,
        description="QA in Indian languages (IndicQA dataset)",
        related_benchmarks=["IndicQA", "TyDiQA-GoldP"],
        failure_modes=["context_loss", "factual_error", "indic_script_error"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        languages=["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"],
        target_tokens=8_000_000,
        priority="critical",
    ),
    "INDIC-TRANS": SkillBucket(
        id="INDIC-TRANS",
        name="Indic Translation",
        category=SkillCategory.INDIC,
        description="Translation between Indian languages and English",
        related_benchmarks=["FLORES-200", "Samanantar", "PMIndia"],
        failure_modes=["translation_error", "indic_script_error"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        languages=[
            "hi",
            "bn",
            "ta",
            "te",
            "mr",
            "gu",
            "kn",
            "ml",
            "pa",
            "or",
            "as",
            "en",
        ],
        target_tokens=10_000_000,
        priority="critical",
    ),
    "INDIC-NLI": SkillBucket(
        id="INDIC-NLI",
        name="Indic Natural Language Inference",
        category=SkillCategory.INDIC,
        description="NLI across Indian languages (Hindi, Bengali, Tamil, etc.)",
        related_benchmarks=["IndicXNLI", "IndicGLUE-NLI"],
        failure_modes=["logic_violation", "indic_morphology_error"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        languages=["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or"],
        target_tokens=5_000_000,
        priority="high",
    ),
    "INDIC-SENT": SkillBucket(
        id="INDIC-SENT",
        name="Indic Sentiment Analysis",
        category=SkillCategory.INDIC,
        description="Sentiment classification in Indian languages",
        related_benchmarks=["IndicGLUE-Sentiment", "IITP-Product"],
        failure_modes=["factual_error", "indic_morphology_error"],
        min_band=Band.B1,
        max_band=Band.B3,
        primary_band=Band.B2,
        stages=[Stage.SFT],
        languages=["hi", "bn", "ta", "te", "mr", "gu"],
        target_tokens=3_000_000,
        priority="high",
    ),
    "INDIC-NER": SkillBucket(
        id="INDIC-NER",
        name="Indic Named Entity Recognition",
        category=SkillCategory.INDIC,
        description="NER in Indian languages",
        related_benchmarks=["IndicGLUE-NER", "WikiANN", "Naamapadam"],
        failure_modes=["extraction_error", "indic_script_error"],
        min_band=Band.B2,
        max_band=Band.B4,
        primary_band=Band.B3,
        stages=[Stage.SFT],
        languages=["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"],
        target_tokens=5_000_000,
        priority="high",
    ),
}


# ================================================================
# LEGACY ALIASES (for backward compatibility)
# Maps old skill names to new canonical names
# ================================================================

SKILL_ALIASES: dict[str, str] = {
    # Reasoning aliases
    "RSN-ARITHMETIC": "RSN-ARITH",
    "RSN-ALGEBRA": "RSN-ALG",
    "RSN-LOGIC": "RSN-LOG",
    "RSN-CAUSAL": "RSN-CAUS",
    "RSN-ANALOGICAL": "RSN-ANAL",
    # Code aliases
    "CODE-COMPLETION": "CODE-GEN-T1",
    "CODE-DEBUG": "CODE-DBG",
    "CODE-EXPLAIN": "CODE-COMP",
    # Language aliases
    "LANG-HINDI": "LANG-HI-COMP",
    "LANG-COHERENCE": "FND-DIS",
    # Knowledge aliases (merged into other categories)
    "KNOW-FACTUAL": "FND-FACT",
    "KNOW-SCIENCE": "RSN-CS",
    "KNOW-COMMONSENSE": "RSN-CS",
}


# ================================================================
# HELPER FUNCTIONS
# ================================================================


def get_skill_bucket(skill_id: str) -> SkillBucket:
    """Get a skill bucket by ID (supports aliases)."""
    # Check for alias first
    canonical_id = SKILL_ALIASES.get(skill_id, skill_id)

    if canonical_id not in SKILL_BUCKETS:
        raise ValueError(f"Unknown skill bucket: {skill_id}")
    return SKILL_BUCKETS[canonical_id]


def get_skills_for_band(band: Band) -> list[SkillBucket]:
    """Get all skills applicable to a band."""
    return [
        s
        for s in SKILL_BUCKETS.values()
        if s.min_band.value <= band.value <= s.max_band.value
    ]


def get_skills_for_stage(stage: Stage) -> list[SkillBucket]:
    """Get all skills applicable to a stage."""
    return [s for s in SKILL_BUCKETS.values() if stage in s.stages]


def get_skills_by_category(category: SkillCategory) -> list[SkillBucket]:
    """Get all skills in a category."""
    return [s for s in SKILL_BUCKETS.values() if s.category == category]


def get_skills_for_language(language: str) -> list[SkillBucket]:
    """Get all skills that support a language."""
    return [s for s in SKILL_BUCKETS.values() if language in s.languages]


def get_critical_skills() -> list[SkillBucket]:
    """Get all critical priority skills."""
    return [s for s in SKILL_BUCKETS.values() if s.priority == "critical"]


def get_failure_mode(mode_id: str) -> FailureMode:
    """Get a failure mode by ID."""
    if mode_id not in FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {mode_id}")
    return FAILURE_MODES[mode_id]


def get_skill_failure_modes(skill_id: str) -> list[FailureMode]:
    """Get all failure modes for a skill."""
    skill = get_skill_bucket(skill_id)
    return [FAILURE_MODES[fm] for fm in skill.failure_modes]


def resolve_skill_alias(skill_id: str) -> str:
    """Resolve a skill alias to its canonical ID."""
    return SKILL_ALIASES.get(skill_id, skill_id)


def list_all_skills() -> list[str]:
    """List all canonical skill IDs."""
    return list(SKILL_BUCKETS.keys())
