import dataclasses
from typing import Dict, Optional, Union
import pathlib
import importlib
import sys

# print(pathlib.Path(__file__))
# Path to the instructions folder
INSTR_PATH = str(pathlib.Path(__file__).parent)
sys.path.append(INSTR_PATH)

import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
    lang
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[lang][instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
    lang
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[lang][instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results, lang):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response, lang)
    out_loose = test_instruction_following_loose(inp, response, lang)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def doc_to_text_qwen_chat(doc):
    # Qwen models expose a tokenizer chat template that uses these sentinel tokens.
    # lm-eval's HF backend is plain-text, so we manually apply a minimal chat wrapper.
    # This tends to matter a lot for instruction-following performance.
    return f"<|im_start|>user\n{doc['prompt']}<|im_end|>\n<|im_start|>assistant\n"

def process_results_en(doc, results):
    return process_results(doc, results, "en")

def process_results_as(doc, results):
    return process_results(doc, results, "as")

def process_results_bn(doc, results):
    return process_results(doc, results, "bn")

def process_results_brx(doc, results):
    return process_results(doc, results, "brx")

def process_results_gu(doc, results):
    return process_results(doc, results, "gu")

def process_results_hi(doc, results):
    return process_results(doc, results, "hi")

def process_results_kn(doc, results):
    return process_results(doc, results, "kn")

def process_results_ml(doc, results):
    return process_results(doc, results, "ml")

def process_results_mr(doc, results):
    return process_results(doc, results, "mr")

def process_results_ne(doc, results):
    return process_results(doc, results, "ne")

def process_results_or(doc, results):
    return process_results(doc, results, "or")

def process_results_pa(doc, results):
    return process_results(doc, results, "pa")

def process_results_sa(doc, results):
    return process_results(doc, results, "sa")

def process_results_sd(doc, results):
    return process_results(doc, results, "sd")

def process_results_ta(doc, results):
    return process_results(doc, results, "ta")

def process_results_te(doc, results):
    return process_results(doc, results, "te")

def process_results_ur(doc, results):
    return process_results(doc, results, "ur")

def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc