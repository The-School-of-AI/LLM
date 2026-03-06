import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TPL_DIR = _ROOT / "experiments" / "6_tokenizer_design_lab" / "tokenizer_validation"
sys.path.insert(0, str(_TPL_DIR))

from sft_template import render_chat, try_split_prompt_answer


def test_try_split_prompt_answer_tab_delim():
    assert try_split_prompt_answer("Q\tA") == ("Q", "A")


def test_try_split_prompt_answer_question_mark_heuristic():
    assert try_split_prompt_answer("What is 2+2? 4") == ("What is 2+2?", "4")


def test_render_chat_contains_role_tokens_and_bos_eos():
    s = render_chat(system="sys", user="u", assistant="a")
    assert s.startswith("<|begin_of_text|>")
    assert "<|system|>" in s
    assert "<|user|>" in s
    assert "<|assistant|>" in s
    assert s.endswith("<|end_of_text|>")
