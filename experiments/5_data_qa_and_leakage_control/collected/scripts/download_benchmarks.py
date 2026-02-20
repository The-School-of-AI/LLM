from datasets import load_dataset
from pathlib import Path
import json

Path("benchmark_registry").mkdir(exist_ok=True)

benchmarks = {
    "mmlu": ("cais/mmlu", "all", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "winogrande": ("winogrande", "winogrande_xl", "validation"),
    "gsm8k": ("gsm8k", "main", "test"),
    "humaneval": ("openai_humaneval", None, "test"),
    "boolq": ("google/boolq", None, "validation"),
    "piqa": ("piqa", None, "validation"),
}

for name, (dataset_id, config, split) in benchmarks.items():
    print(f"\n📥 Downloading {name}...")
    
    try:
        if config:
            ds = load_dataset(dataset_id, config, split=split)
        else:
            ds = load_dataset(dataset_id, split=split)
        
        output_file = f"benchmark_registry/{name}_test.jsonl"
        
        with open(output_file, 'w') as f:
            for item in ds:
                if 'question' in item:
                    text = item['question']
                elif 'prompt' in item:
                    text = item['prompt']
                elif 'sentence' in item:
                    text = item['sentence']
                elif 'ctx' in item:
                    text = item['ctx']
                else:
                    text = str(item)
                
                f.write(json.dumps({"question": text, "source": name}) + '\n')
        
        print(f"✅ {name}: {len(ds)} samples")
    
    except Exception as e:
        print(f"❌ {name}: {e}")

print("\n✅ Done!")