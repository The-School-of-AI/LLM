import argparse
import json
import os
import sys

import torch
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_model_args(args_str):
    """
    Parses a string like "pretrained=HuggingFaceTB/SmolLM2-135M,dtype=float16" into a dict.
    """
    args = {}
    if not args_str:
        return args
    for part in args_str.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip()
    return args


def generate_one_completion(model, tokenizer, prompt, device):
    """
    Generates a single completion for a given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # We define a stop sequence which is important for code generation to stop it from generating infinite loops or extra functions
    # HumanEval usually stops at "\nclass", "\ndef", "\nif", "\nprint" sometimes, but standard is often just letting it generate
    # until max tokens or EOS, and then post-processing.
    # However, 'human-eval' evaluation script is robust to some extra text, but cleaner is better.
    # For simplicity in this script, we'll let it generate and trust the eos_token or max_new_tokens.
    # Improvements could involve StoppingCriteria.

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Pass@1 usually uses greedy decoding or low temperature
            temperature=0.0,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    # Simple decoding
    tokenizer.decode(outputs[0], skip_special_tokens=True)

    # We only want the *new* generated code, but HumanEval prompts include the function signature,
    # so often we need the full function for context or just the body.
    # The 'human_eval' package expects 'completion' to be the part *after* the prompt usually,
    # OR the full code if passing to a robust evaluator.
    # Default 'evaluate_functional_correctness' combines prompt + completion.
    # So we should return ONLY the completion part.

    # inputs['input_ids'].shape[1] is the length of prompt
    prompt_len = inputs["input_ids"].shape[1]
    completion_tokens = outputs[0][prompt_len:]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_args", required=True, help="Model arguments e.g. pretrained=gpt2"
    )
    # eval_runner might pass other args, catch them to avoid error
    parser.add_argument("--batch_size", default="1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", default=None, type=int)
    args, unknown = parser.parse_known_args()

    model_args_dict = parse_model_args(args.model_args)
    model_name = model_args_dict.get("pretrained")

    if not model_name:
        # Fallback or error
        print(
            json.dumps(
                {
                    "name": "HumanEval",
                    "status": "failed",
                    "error": "No pretrained model specified in model_args",
                }
            )
        )
        return

    # Device selection
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Handle models that don't have a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
            device
        )
        model.eval()
    except Exception as e:
        print(
            json.dumps(
                {
                    "name": "HumanEval",
                    "status": "failed",
                    "error": f"Model load failed: {str(e)}",
                }
            )
        )
        return

    # Read problems
    problems = read_problems()

    # Limit for testing
    task_ids = list(problems.keys())
    if args.limit:
        task_ids = task_ids[: args.limit]

    samples = []

    # Generation
    # Note: Not using Batching for simplicity and correctness first
    for task_id in tqdm(task_ids, desc="Generating HumanEval"):
        prompt = problems[task_id]["prompt"]
        try:
            completion = generate_one_completion(model, tokenizer, prompt, device)
            samples.append(dict(task_id=task_id, completion=completion))
        except Exception as e:
            # If generation fails for one, log/skip
            print(f"Error generating completion for task {task_id}: {e}")
            continue

    # Write samples to temporary file
    # Use generic temp name or specific one
    samples_file = f"humaneval_samples_{os.getpid()}.jsonl"
    write_jsonl(samples_file, samples)

    # For subset evaluation, we need to create a temporary problem file containing only the sampled tasks
    # otherwise human_eval complains "Some problems are not attempted"
    subset_problems_file = f"humaneval_problems_{os.getpid()}.jsonl"
    subset_problems = [problems[tid] for tid in task_ids]
    write_jsonl(subset_problems_file, subset_problems)

    # Context manager to suppress stdout/stderr from external libs
    class SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    try:
        # Evaluate
        # k=[1] means calculating pass@1
        # n_workers=1 to avoid multiprocessing issues in some envs, or default (4)
        # We pass problem_file to restrict valid tasks to the ones we generated
        with SuppressOutput():
            results = evaluate_functional_correctness(
                samples_file, k=[1], n_workers=1, problem_file=subset_problems_file
            )

        pass_at_1 = results.get("pass@1", 0.0)

        output = {
            "name": "HumanEval",
            "status": "success",
            "score": pass_at_1,
            "metrics": results,
        }
    except Exception as e:
        output = {
            "name": "HumanEval",
            "status": "failed",
            "error": f"Evaluation failed: {str(e)}",
        }
    finally:
        # Cleanup
        if os.path.exists(samples_file):
            os.remove(samples_file)
        if os.path.exists(subset_problems_file):
            os.remove(subset_problems_file)

    # Print JSON to stdout for eval_runner to capture
    print(json.dumps(output))


if __name__ == "__main__":
    main()
