import logging
from collections import defaultdict

from coreset_engine.ingestion.local import JsonlDataSource
from coreset_engine.scoring.perplexity import DifficultyScorer
from coreset_engine.filters.dedup import DedupRegister
from coreset_engine.selection.curriculum import CurriculumLoader
from coreset_engine.selection.sampler import StratifiedSampler
from coreset_engine.output.manifest import ManifestWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoresetBuilder:
    def __init__(self, config_path: str, data_path: str, output_dir: str):
        self.curriculum = CurriculumLoader(config_path).load() # Now returns List[Dict]
        self.data_source = JsonlDataSource(data_path)
        self.scorer = DifficultyScorer()
        self.dedup = DedupRegister()
        self.sampler = StratifiedSampler(self.curriculum)
        self.writer = ManifestWriter(output_dir)

    def build(self):
        logging.info("Starting Coreset Build Process...")
        
        # 1. Ingestion & Scoring & Dedup Pass
        all_items = []
        logging.info("Pass 1: Ingestion, Scoring and Deduplication...")
        
        count = 0
        dropped_dedup = 0
        
        for record in self.data_source:
            # deduplication
            text = record.get('text', '')
            if self.dedup.is_duplicate(text):
                dropped_dedup += 1
                continue
            
            # Scoring
            score = self.scorer.score(record)
            record['difficulty_score'] = score
            
            # Keep metadata only in memory (drop big text)
            meta_record = {k:v for k,v in record.items() if k != 'text'}
            all_items.append(meta_record)
            
            count += 1
            if count % 10000 == 0:
                logging.info(f"Processed {count} items. Dedup dropped: {dropped_dedup}")

        logging.info(f"Pass 1 Complete. Valid Items: {len(all_items)}. Dedup Dropped: {dropped_dedup}")
        
        # 2. Sampling / Selection
        logging.info("Pass 2: Partitioning stages...")
        stage_outputs = self.sampler.partition(all_items)
        
        # 3. Writing Output
        logging.info("Pass 3: Writing Manifests...")
        for stage_id, items in stage_outputs.items():
            # Calc summary stats
            total_tokens = sum(x['token_count'] for x in items)
            # Distribution: Modality and Band
            dist = defaultdict(int)
            for x in items:
                k = f"{x.get('assigned_band', '?')}_{x.get('assigned_modality', '?')}"
                dist[k] += x['token_count']
            
            summary = {
                "stage": stage_id,
                "total_tokens": total_tokens,
                "distribution": dict(dist)
            }
            
            self.writer.write(stage_id, items, summary)
            
        logging.info("Build Complete.")
