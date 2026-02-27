"""
metrics_calculator_single.py
============================
Standalone Python module for robust modality scoring and curriculum band assignment on a single text snippet.
"""
import re
from typing import Dict, Any

# -----------------------------
# V5 PATTERN DEFINITIONS (Regex)
# -----------------------------
AGENTIC_STRUCTURAL_PATTERN = r"""(?x)
    (?:(?:Step\s+\d+|Task\s+\d+):\s*(?:Call|Execute|Run|Use|Invoke)\s+\w+)|
    (?:(?:tool|function|api)_(?:use|call|invoke)\s*\()|
    (?:\[(?:PLAN|ACTION|TOOL|STEP)\s*\d*\])|
    (?:Thought\s*\d*:.*{10,}Action\s*\d*:)
"""
AGENTIC_VOCAB_PATTERN = r'\b(?:execute|invoke|call|dispatch|orchestrate|coordinate|delegate|subgoal|subtask|decompose|breakdown|workflow|pipeline)\b'
COT_EXPLICIT_PATTERN = r"""(?x)
    (?:Let's\s+think\s+(?:step[- ]by[- ]step|through\s+this|carefully|systematically))|
    (?:\[(?:REASONING|THINKING|ANALYSIS)\])|
    (?:I\s+(?:need\s+to|should|must)\s+(?:think\s+about|consider|analyze))
"""
COT_REASONING_CONNECTIVES = r'\b(?:therefore|thus|hence|because|since|this\s+means|which\s+implies)\b'
EDUCATIONAL_MARKER_PATTERN = r"(?i)###\s*(?:Explanation|Question|Answer|Topic|Metadata|Prerequisites):"
FORMAL_REASONING_PATTERN = r"""(?x)
    (?:Proof:|Theorem:|Lemma:|Corollary:)|
    (?:Q\.E\.D\.|∎|□)|
    (?:(?:By|Using)\s+(?:induction|contradiction|construction))|
    (?:It\s+follows\s+that|We\s+can\s+deduce|This\s+implies)
"""
MATH_SYMBOLS_PATTERN = r'[∀∃∈∉⊂⊆∪∩∅⇒⇔∧∨¬→↔⊢⊨≡≠≤≥±∓∞∑∏∫√]'
TABLE_ROW_PATTERN = r'\|(?:\s*\w+\s*\|){2,}'
TABLE_HEADER_KEYWORDS = r'(?i)\b(?:name|id|value|type|date|count|total|column|field|description)\b'
MARKDOWN_TABLE_SEPARATOR = r'\|[-:]+\|[-:]+\|'
PYTHON_SYNTAX = r'(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+\w+'
JAVASCRIPT_SYNTAX = r'(?:function\s+\w+\s*\(|const|let|var)\s+\w+\s*=|=>'
JAVA_CPP_SYNTAX = r'(?:public|private|protected|#include|int\s+main)'
CODE_STRUCTURE = r'^\s{2,}\S'
CODE_SYNTAX_CHARS = r'[;{}()\[\]]'
CAMEL_SNAKE_CASE = r'\b[a-z]+[A-Z]\w+\b|\b[a-z]+_[a-z_]+\b'
EQUATION_PATTERN = r'[a-z]\s*[+\-*/=]\s*[a-z0-9]|[a-z]\^[0-9]|\([a-z0-9\s+\-*/]+\)\s*='
LATEX_COMMANDS = r'\\(?:frac|sum|prod|int|lim|infty|sqrt|cdot|times|begin\{equation)'
MATH_TERMINOLOGY = r'\b(?:theorem|lemma|proof|equation|derivative|integral|matrix|vector|polynomial)\b'

# -----------------------------
# Helper Functions
# -----------------------------
def basic_metrics(text: str) -> Dict[str, Any]:
    byte_length = len(text.encode('utf-8'))
    char_length = len(text)
    word_count = len(re.findall(r'\S+', text))
    line_count = text.count('\n') + 1
    token_count_estimate = int(word_count * 1.3)
    fertility_estimate = char_length / token_count_estimate if token_count_estimate > 0 else 1.0
    return {
        'byte_length': byte_length,
        'char_length': char_length,
        'word_count': word_count,
        'line_count': line_count,
        'token_count_estimate': token_count_estimate,
        'fertility_estimate': fertility_estimate,
    }

def robust_modality_scores(text: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    word_count = metrics['word_count']
    char_length = metrics['char_length']
    line_count = metrics['line_count']
    # Agentic
    agentic_score = 0
    if len(re.findall(AGENTIC_STRUCTURAL_PATTERN, text)) >= 2:
        agentic_score += 3
    if len(re.findall(AGENTIC_VOCAB_PATTERN, text)) / max(word_count, 1) > 0.006:
        agentic_score += 2
    if len(re.findall(r'\b(?:subgoal|subtask|decompose|breakdown|workflow|pipeline)\b', text)) >= 3:
        agentic_score += 2
    if len(re.findall(r'(?:def|function)\s+\w+_(?:tool|call|agent)', text)) >= 1:
        agentic_score += 2
    is_agentic = int(agentic_score >= 5)
    # CoT
    cot_score = 0
    if len(re.findall(COT_EXPLICIT_PATTERN, text)) >= 1:
        cot_score += 3
    if len(re.findall(COT_REASONING_CONNECTIVES, text)) / max(word_count, 1) > 0.02:
        cot_score += 2
    if len(re.findall(r'\?\s+.{30,}\.', text)) >= 2 and len(re.findall(r'(?:So|Therefore|Thus),?\s+', text)) >= 2:
        cot_score += 2
    if len(re.findall(EDUCATIONAL_MARKER_PATTERN, text)) >= 3:
        cot_score += 3
    is_cot = int(cot_score >= 5)
    # Reasoning
    reasoning_score = 0
    if len(re.findall(FORMAL_REASONING_PATTERN, text)) >= 2:
        reasoning_score += 4
    math_symbols_count = len(re.findall(MATH_SYMBOLS_PATTERN, text))
    if math_symbols_count / max(char_length, 100) > 0.01:
        reasoning_score += 3
    if len(re.findall(r'(?:If|Suppose|Assume).{10,},\s+then', text)) >= 3:
        reasoning_score += 2
    is_reasoning = int(reasoning_score >= 6)
    # Table
    table_lines = sum(1 for line in text.split('\n') if len(re.findall(r'\|', line)) >= 2)
    table_score = 0
    if table_lines >= 3 and table_lines / max(line_count, 1) >= 0.5:
        table_score += 3
    if len(re.findall(TABLE_ROW_PATTERN, text)) >= 3:
        table_score += 2
    if len(re.findall(TABLE_HEADER_KEYWORDS, text[:200])) >= 2:
        table_score += 2
    if len(re.findall(MARKDOWN_TABLE_SEPARATOR, text)) >= 1:
        table_score += 3
    is_table = int(table_score >= 5)
    # Code
    code_score = 0
    if len(re.findall(PYTHON_SYNTAX, text)) >= 2:
        code_score += 6
    if len(re.findall(JAVASCRIPT_SYNTAX, text)) >= 2:
        code_score += 6
    if len(re.findall(JAVA_CPP_SYNTAX, text)) >= 2:
        code_score += 6
    if len(re.findall(CODE_STRUCTURE, text)) >= 5 or len(re.findall(CODE_SYNTAX_CHARS, text)) >= 10:
        code_score += 3
    if len(re.findall(CAMEL_SNAKE_CASE, text)) >= 5:
        code_score += 2
    is_code = int(code_score >= 10)
    # Math
    math_score = 0
    if len(re.findall(EQUATION_PATTERN, text)) >= 2:
        math_score += 3
    if len(re.findall(LATEX_COMMANDS, text)) >= 2:
        math_score += 3
    if len(re.findall(MATH_TERMINOLOGY, text)) >= 2:
        math_score += 2
    is_math = int(math_score >= 5)
    return {
        'agentic_score': agentic_score,
        'is_agentic': is_agentic,
        'cot_score': cot_score,
        'is_cot': is_cot,
        'reasoning_score': reasoning_score,
        'is_reasoning': is_reasoning,
        'table_score': table_score,
        'is_table': is_table,
        'code_score': code_score,
        'is_code': is_code,
        'math_score': math_score,
        'is_math': is_math,
    }

def assign_curriculum_band(modality_scores: Dict[str, Any]) -> str:
    # Simple band assignment based on scores (example logic)
    score_sum = sum([
        modality_scores['agentic_score'],
        modality_scores['cot_score'],
        modality_scores['reasoning_score'],
        modality_scores['table_score'],
        modality_scores['code_score'],
        modality_scores['math_score'],
    ])
    if score_sum >= 20:
        return 'B5'
    elif score_sum >= 15:
        return 'B4'
    elif score_sum >= 10:
        return 'B3'
    elif score_sum >= 5:
        return 'B2'
    elif score_sum >= 2:
        return 'B1'
    else:
        return 'B0'

def analyze_text(text: str) -> Dict[str, Any]:
    metrics = basic_metrics(text)
    modality_scores = robust_modality_scores(text, metrics)
    band = assign_curriculum_band(modality_scores)
    return {
        'metrics': metrics,
        'modality_scores': modality_scores,
        'band': band,
    }

# Example usage:
if __name__ == '__main__':
    sample_text = """Let's think step by step. Proof: Suppose x = 2, then y = 4. def my_function(): pass | Name | Value |\n"""
    result = analyze_text(sample_text)
    print(result)
