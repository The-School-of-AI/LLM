from typing import Dict, Any, Tuple

class BucketMapper:
    """
    Maps a data record to a canonical bucket: (Band, Modality).
    """
    def __init__(self):
        # Heuristic thresholds for Zlib Compression Ratio (proxy for perplexity)
        # Low ratio (high compression) = Easy = B0
        # High ratio (low compression) = Hard = B5
        # These are illustrative and should be calibrated.
        # B0: 0.0 - 0.25 (Very repetitive)
        # B1: 0.25 - 0.35
        # B2: 0.35 - 0.45
        # B3: 0.45 - 0.55
        # B4: 0.55 - 0.70
        # B5: 0.70 - 1.0 (High entropy / Code / Math)
        self.band_thresholds = [
            (0.25, "B0"),
            (0.35, "B1"),
            (0.45, "B2"),
            (0.55, "B3"),
            (0.70, "B4"),
            (1.01, "B5") 
        ]

    def get_bucket(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """
        Returns (band_id, modality_id)
        """
        score = record.get('difficulty_score', 0.0)
        
        # Determine Band
        band = "B0"
        for thresh, b_name in self.band_thresholds:
            if score < thresh:
                band = b_name
                break
        
        # Determine Modality
        # Rely on metadata 'modality' or 'domain' tags. 
        # Default to 'general_text' if missing.
        modality = record.get('modality', 'general_text')
        
        # Fallback/Normalization if needed
        valid_modalities = {"general_text", "code", "cot_reasoning", "agentic_traces"}
        if modality not in valid_modalities:
            # Map known domains to modalities?
            domain = record.get('domain', '')
            if 'code' in domain:
                modality = 'code'
            elif 'math' in domain:
                modality = 'cot_reasoning' # Just a guess for now
            else:
                modality = 'general_text'
                
        return band, modality
