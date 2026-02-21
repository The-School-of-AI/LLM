"""Detection layers - N-gram, MinHash, and Semantic"""

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
    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.keys = {}  # key -> {"text": str, "minhash": MinHash}

    def build_index(self, registry):
        console.print("[yellow]Building MinHash index...[/yellow]")

        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)

            for idx, text in enumerate(texts):
                mh = self._hash(text)
                key = f"{name}_{idx}"
                self.lsh.insert(key, mh)
                self.keys[key] = {"text": text[:100], "minhash": mh}

            console.print(f"✓ {name}: {len(texts)} hashes")

        console.print(f"[green]✓ MinHash index ready[/green]\n")

    def _hash(self, text):
        mh = MinHash(num_perm=self.num_perm)
        words = text.split()
        for i in range(len(words) - 1):
            shingle = ' '.join(words[i:i+2])
            mh.update(shingle.encode('utf-8'))
        return mh

    def scan(self, texts):
        matches = defaultdict(list)

        for idx, text in enumerate(tqdm(texts, desc="MinHash")):
            mh = self._hash(text)
            similar = self.lsh.query(mh)

            if similar:
                for key in similar:
                    benchmark = key.rsplit('_', 1)[0]
                    jaccard = mh.jaccard(self.keys[key]["minhash"])
                    if jaccard < self.threshold:
                        continue  # LSH false positive candidate, discard
                    matches[benchmark].append({
                        "idx": idx,
                        "text": text[:150],
                        "match": self.keys[key]["text"],
                        "jaccard": round(jaccard, 3)
                    })

        return dict(matches)


class SemanticDetector:
    def __init__(self, threshold=0.9, model_name="all-MiniLM-L6-v2", batch_size=512):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install faiss-cpu sentence-transformers")

        self.threshold = threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []  # meta[i] = {"benchmark": str, "text": str} — parallel to FAISS index

    def build_index(self, registry):
        import faiss

        dim = self.model.get_sentence_embedding_dimension()
        console.print(f"[yellow]Building semantic index ({dim}d)...[/yellow]")

        all_texts, all_meta = [], []
        for name in registry.benchmarks.keys():
            for text in registry.get_texts(name):
                all_texts.append(text)
                all_meta.append({"benchmark": name, "text": text[:100]})

        embeddings = self.model.encode(
            all_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
        self.index.add(embeddings)
        self.meta = all_meta

        console.print(f"[green]✓ Semantic index ready: {len(all_texts)} vectors[/green]\n")

    def scan(self, texts):
        matches = defaultdict(list)

        for batch_start in tqdm(range(0, len(texts), self.batch_size), desc="Semantic"):
            batch = texts[batch_start:batch_start + self.batch_size]

            embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")

            scores, indices = self.index.search(embeddings, k=1)

            for i, (score, matched_idx) in enumerate(zip(scores[:, 0], indices[:, 0])):
                if float(score) >= self.threshold:
                    meta = self.meta[int(matched_idx)]
                    global_idx = batch_start + i
                    matches[meta["benchmark"]].append({
                        "idx": global_idx,
                        "text": texts[global_idx][:150],
                        "match": meta["text"],
                        "cosine": round(float(score), 3)
                    })

        return dict(matches)