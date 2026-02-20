"""Detection layers - N-gram and MinHash"""

from collections import defaultdict
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from rich.console import Console

console = Console()


class NGramDetector:
    def __init__(self, n=13):
        self.n = n
        self.index = {}
    
    def build_index(self, registry):
        console.print(f"[yellow]Building {self.n}-gram index...[/yellow]")
        
        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)
            ngrams = set()
            
            for text in texts:
                ngrams.update(self._extract(text))
            
            self.index[name] = ngrams
            console.print(f"✓ {name}: {len(ngrams)} n-grams")
        
        console.print(f"[green]✓ N-gram index ready[/green]\n")
    
    def _extract(self, text):
        words = text.split()
        return [' '.join(words[i:i+self.n]) for i in range(len(words)-self.n+1)]
    
    def scan(self, texts):
        matches = defaultdict(list)
        
        for idx, text in enumerate(tqdm(texts, desc="N-gram")):
            text_ngrams = set(self._extract(text))
            
            for benchmark, bench_ngrams in self.index.items():
                overlap = text_ngrams & bench_ngrams
                if overlap:
                    matches[benchmark].append({
                        "idx": idx,
                        "text": text[:150],
                        "count": len(overlap)
                    })
        
        return dict(matches)


class MinHashDetector:
    def __init__(self, threshold=0.6, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.keys = {}
    
    def build_index(self, registry):
        console.print("[yellow]Building MinHash index...[/yellow]")
        
        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)
            
            for idx, text in enumerate(texts):
                mh = self._hash(text)
                key = f"{name}_{idx}"
                self.lsh.insert(key, mh)
                self.keys[key] = text[:100]
            
            console.print(f"✓ {name}: {len(texts)} hashes")
        
        console.print(f"[green]✓ MinHash index ready[/green]\n")
    
    def _hash(self, text):
        mh = MinHash(num_perm=self.num_perm)
        for i in range(len(text)-2):
            mh.update(text[i:i+3].encode('utf-8'))
        return mh
    
    def scan(self, texts):
        matches = defaultdict(list)
        
        for idx, text in enumerate(tqdm(texts, desc="MinHash")):
            mh = self._hash(text)
            similar = self.lsh.query(mh)
            
            if similar:
                for key in similar:
                    benchmark = key.rsplit('_', 1)[0]
                    matches[benchmark].append({
                        "idx": idx,
                        "text": text[:150],
                        "match": self.keys[key]
                    })
        
        return dict(matches)