from typing import List, Dict, Any
import random
from collections import defaultdict
from .bucketer import BucketMapper

class StratifiedSampler:
    """
    Selects items for each stage based on Production Curriculum (Band + Modality).
    """
    def __init__(self, flat_stages: List[Dict[str, Any]]):
        self.stages = flat_stages
        self.bucketer = BucketMapper()
        
    def partition(self, all_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partitions items into stages.
        """
        # 1. Bucketize all items
        # Structure: bucket_pool[(band, modality)] = [items...]
        bucket_pool = defaultdict(list)
        
        for item in all_items:
            # We assume 'difficulty_score' is already populated
            band, modality = self.bucketer.get_bucket(item)
            item['assigned_band'] = band
            item['assigned_modality'] = modality
            bucket_pool[(band, modality)].append(item)

        # Shuffle each bucket once to ensure random sampling
        for k in bucket_pool:
            random.shuffle(bucket_pool[k])

        stage_outputs = {}
        
        # Track globally consumed item IDs to prevent overlap
        consumed_ids = set()

        for stage_config in self.stages:
            stage_id = stage_config['stage_id']
            target_tokens = stage_config['target_tokens']
            band_weights = stage_config['band_weights']
            modality_weights = stage_config['modality_weights']
            
            selected_items = []

            # Strategy:
            # We fundamentally stratify by BAND first (Difficulty is the primary constraint).
            # Within each Band, we try to respect Modality weights if possible.
            # But the 'modality_weights' in the config are GLOBAL for the stage.
            # This is a matrix constraints problem. 
            # Simplified Greedy Approach:
            # 1. Calculate target tokens for each Band (e.g. B0: 30% of Total).
            # 2. Within that Band allowance, try to pick Modalities roughly proportional to global modality weights?
            # Actually, `curriculum.yaml` provides Band weights AND Modality weights separately.
            # Ideally they should be satisfied jointly.
            # Let's iterate Bans. For Band `b`, we need `T_b = Total * Weight_b`.
            # We fill `T_b` by sampling from `(b, m)` buckets.
            # We can prioritize modalities that are "under-represented" relative to the target modality weights.
            
            for band, b_weight in band_weights.items():
                if b_weight <= 0:
                    continue
                
                band_target = int(target_tokens * b_weight)
                band_collected = 0
                
                # Available buckets for this band
                # e.g. (B0, general_text), (B0, code), ...
                # We need to draw from these.
                
                # Check available tokens in this band (across all modalities)
                # Filter out consumed
                
                while band_collected < band_target:
                    # Pick a modality to sample from next
                    # We pick a modality that exists in this band and has items left
                    candidate_modalities = []
                    for (b, m), items in bucket_pool.items():
                        if b == band and items:
                             candidate_modalities.append(m)
                    
                    if not candidate_modalities:
                        break # Band exhausted
                    
                    # Round-robin or weighted random choice among modalities?
                    # Let's use weighted choice based on stage's modality_weights
                    weights = [modality_weights.get(m, 0.01) for m in candidate_modalities]
                    chosen_modality = random.choices(candidate_modalities, weights=weights, k=1)[0]
                    
                    # Pop item
                    # (Note: Efficient popping from end of list is O(1))
                    item = bucket_pool[(band, chosen_modality)].pop()
                    
                    if id(item) in consumed_ids:
                        continue # Should not happen if we remove from pool, but safety check
                        
                    selected_items.append(item)
                    consumed_ids.add(id(item))
                    band_collected += item['token_count']
                    
            stage_outputs[stage_id] = selected_items
            
        return stage_outputs
