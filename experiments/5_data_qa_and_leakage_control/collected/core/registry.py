"""Benchmark registry - loads and manages benchmark datasets"""

import json
from pathlib import Path
from rich.console import Console

console = Console()


class BenchmarkRegistry:
    def __init__(self, benchmarks_path="benchmarks"):
        self.benchmarks_path = Path(benchmarks_path)
        self.benchmarks = {}
        
    def load_all(self):
        console.print("[yellow]Loading benchmarks...[/yellow]")
        
        for file in self.benchmarks_path.glob("*_test.jsonl"):
            name = file.stem
            data = []
            
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            self.benchmarks[name] = data
            console.print(f"✓ {name}: {len(data)} samples")
        
        console.print(f"[green]✓ Total: {len(self.benchmarks)} benchmarks[/green]\n")
        return self
    
    def get_texts(self, benchmark_name):
        texts = []
        for sample in self.benchmarks.get(benchmark_name, []):
            text = sample.get('question', str(sample))
            texts.append(self._normalize(text))
        return texts
    
    def _normalize(self, text):
        return ' '.join(str(text).lower().strip().split())