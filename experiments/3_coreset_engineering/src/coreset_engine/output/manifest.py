import json
import os
from typing import List, Dict, Any

class ManifestWriter:
    """
    Writes the selected coreset indices to disk.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def write(self, stage_id: str, items: List[Dict[str, Any]], summary: Dict[str, Any]):
        """
        Writes two files:
        1. {stage_id}_index.jsonl: The actual index
        2. {stage_id}_manifest.json: Summary metadata
        """
        index_path = os.path.join(self.output_dir, f"{stage_id}_index.jsonl")
        manifest_path = os.path.join(self.output_dir, f"{stage_id}_manifest.json")
        
        # Write Index
        with open(index_path, 'w', encoding='utf-8') as f:
            for item in items:
                # Minimal record for dataloader
                out_record = {
                    "original_file": item['file_path'],
                    "line_number": item['file_line'],
                    "token_count": item['token_count'],
                    "domain": item['domain'],
                    "difficulty_score": item.get('difficulty_score'),
                    "assigned_band": item.get('assigned_band'),
                    "assigned_modality": item.get('assigned_modality')
                }
                f.write(json.dumps(out_record) + "\n")
        
        # Write Manifest
        summary['total_items'] = len(items)
        summary['output_file'] = os.path.abspath(index_path)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Written stage {stage_id} with {len(items)} items to {index_path}")
