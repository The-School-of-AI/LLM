#!/usr/bin/env python3
"""
Side-by-Side Generation Comparison
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Loads a base model and an optional SFT adapter to compare outputs on a set of prompts.
"""

from qlora_config import create_argument_parser, load_config

def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Use config values, with CLI overrides already handled by load_config
    base_model = config.generation.base_model_path
    adapter_path = config.generation.adapter_path
    prompts = config.generation.prompts
    max_new_tokens = config.generation.max_new_tokens
    temperature = config.generation.temperature

    if not base_model:
        print("Error: Base model path must be specified (config or --gen_base_model)")
        return

    print(f"Loading base model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    results = []
    print("\nRunning Base Model Generations...")
    for prompt in prompts:
        results.append({"prompt": prompt, "base": generate(model, tokenizer, prompt, max_new_tokens, temperature)})

    if adapter_path:
        print(f"\nLoading LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Running SFT Model Generations...")
        for i, prompt in enumerate(prompts):
            results[i]["sft"] = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    print("\n" + "="*80)
    print(f"{'PROMPT':<40} | {'BASE RESPONSE':<40} | {'SFT RESPONSE':<40}")
    print("="*80)
    for res in results:
        print(f"{res['prompt'][:37]+'...':<40} | {res['base'][:37].replace(chr(10), ' ')+'...':<40} | {res.get('sft', 'N/A')[:37].replace(chr(10), ' ')+'...':<40}")
    
    print("\nDetailed comparison saved to comparison_results.txt")
    with open("comparison_results.txt", "w") as f:
        for res in results:
            f.write(f"PROMPT: {res['prompt']}\n")
            f.write(f"{'-'*20}\n")
            f.write(f"BASE:\n{res['base']}\n")
            if "sft" in res:
                f.write(f"\nSFT:\n{res['sft']}\n")
            f.write(f"\n{'='*80}\n\n")

if __name__ == "__main__":
    main()
