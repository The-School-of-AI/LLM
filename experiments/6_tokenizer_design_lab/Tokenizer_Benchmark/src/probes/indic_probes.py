"""
Indic Probe Generator - Synthetic Indic language patterns.

Generates synthetic Indic script content for tokenizer evaluation.
Tests tokenizer handling of:
- Devanagari (Hindi, Sanskrit, Marathi)
- Dravidian scripts (Tamil, Telugu, Kannada, Malayalam)
- Mixed English-Indic text
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class IndicProbe:
    """A single Indic language probe."""
    content: str
    script: str
    category: str  # pure, mixed, technical
    difficulty: str


class IndicProbeGenerator:
    """
    Generates synthetic Indic language probes.
    
    Creates pattern-based Indic text to test tokenizer behavior
    on non-Latin scripts WITHOUT using real benchmark content.
    """
    
    # Devanagari character ranges and patterns
    DEVANAGARI = {
        "vowels": "अआइईउऊऋएऐओऔ",
        "consonants": "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
        "matras": "ािीुूेैोौंःँ",
        "numbers": "०१२३४५६७८९",
        "punctuation": "।॥",
    }
    
    # Tamil character ranges
    TAMIL = {
        "vowels": "அஆஇஈஉஊஎஏஐஒஓஔ",
        "consonants": "கஙசஞடணதநபமயரலவழளறன",
        "matras": "ாிீுூெேைொோௌ",
        "numbers": "௦௧௨௩௪௫௬௭௮௯",
    }
    
    # Telugu character ranges
    TELUGU = {
        "vowels": "అఆఇఈఉఊఋఎఏఐఒఓఔ",
        "consonants": "కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహ",
        "matras": "ాిీుూెేైొోౌంః",
        "numbers": "౦౧౨౩౪౫౬౭౮౯",
    }
    
    # Kannada character ranges
    KANNADA = {
        "vowels": "ಅಆಇಈಉಊಋಎಏಐಒಓಔ",
        "consonants": "ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹ",
        "matras": "ಾಿೀುೂೆೇೈೊೋೌಂಃ",
        "numbers": "೦೧೨೩೪೫೬೭೮೯",
    }
    
    # Malayalam character ranges
    MALAYALAM = {
        "vowels": "അആഇഈഉഊഋഎഏഐഒഓഔ",
        "consonants": "കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹ",
        "matras": "ാിീുൂെേൈൊോൗംഃ",
        "numbers": "൦൧൨൩൪൫൬൭൮൯",
    }
    
    # Common word patterns (syllable structures)
    SYLLABLE_PATTERNS = [
        "CV",    # Consonant + Vowel/Matra
        "CVC",   # Consonant + Vowel + Consonant
        "CCV",   # Conjunct + Vowel
        "V",     # Just vowel
        "VC",    # Vowel + Consonant
    ]
    
    # English technical terms (for mixed content)
    TECH_TERMS = [
        "algorithm", "database", "function", "variable", "parameter",
        "API", "JSON", "HTTP", "machine learning", "neural network",
        "data", "software", "hardware", "server", "client",
        "Python", "JavaScript", "model", "training", "inference",
    ]
    
    SCRIPTS = {
        "devanagari": DEVANAGARI,
        "tamil": TAMIL,
        "telugu": TELUGU,
        "kannada": KANNADA,
        "malayalam": MALAYALAM,
    }
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _generate_syllable(self, script_data: Dict[str, str]) -> str:
        """Generate a single syllable."""
        pattern = self.rng.choice(self.SYLLABLE_PATTERNS)
        syllable = ""
        
        for char_type in pattern:
            if char_type == 'C':
                syllable += self.rng.choice(script_data["consonants"])
            elif char_type == 'V':
                if syllable and self.rng.random() < 0.7:
                    # Use matra after consonant
                    syllable += self.rng.choice(script_data["matras"])
                else:
                    syllable += self.rng.choice(script_data["vowels"])
        
        return syllable
    
    def _generate_word(self, script_data: Dict[str, str], syllables: int = 2) -> str:
        """Generate a synthetic word."""
        return "".join(self._generate_syllable(script_data) for _ in range(syllables))
    
    def _generate_sentence(self, script_data: Dict[str, str], words: int = 5) -> str:
        """Generate a synthetic sentence."""
        word_list = []
        for _ in range(words):
            syllable_count = self.rng.randint(1, 4)
            word_list.append(self._generate_word(script_data, syllable_count))
        
        sentence = " ".join(word_list)
        
        # Add punctuation
        if "punctuation" in script_data:
            sentence += self.rng.choice(script_data["punctuation"])
        else:
            sentence += "."
        
        return sentence
    
    def _generate_number(self, script_data: Dict[str, str], digits: int = 3) -> str:
        """Generate a number in the script's numerals."""
        if "numbers" in script_data and self.rng.random() < 0.5:
            return "".join(self.rng.choice(script_data["numbers"]) for _ in range(digits))
        else:
            return str(self.rng.randint(10**(digits-1), 10**digits - 1))
    
    def generate_pure_probe(self, script: str = "devanagari", difficulty: str = "medium") -> IndicProbe:
        """Generate a pure Indic script probe."""
        script_data = self.SCRIPTS.get(script, self.DEVANAGARI)
        
        if difficulty == "easy":
            sentences = 1
            words_per_sentence = 3
        elif difficulty == "medium":
            sentences = 2
            words_per_sentence = 5
        else:
            sentences = 4
            words_per_sentence = 7
        
        content = " ".join(
            self._generate_sentence(script_data, words_per_sentence)
            for _ in range(sentences)
        )
        
        return IndicProbe(content=content, script=script, category="pure", difficulty=difficulty)
    
    def generate_mixed_probe(self, script: str = "devanagari", difficulty: str = "medium") -> IndicProbe:
        """Generate a mixed English-Indic probe."""
        script_data = self.SCRIPTS.get(script, self.DEVANAGARI)
        
        parts = []
        num_parts = 4 if difficulty == "easy" else 6 if difficulty == "medium" else 8
        
        for i in range(num_parts):
            if i % 2 == 0:
                # Indic part
                parts.append(self._generate_word(script_data, self.rng.randint(1, 3)))
            else:
                # English part
                parts.append(self.rng.choice(self.TECH_TERMS))
        
        content = " ".join(parts)
        
        return IndicProbe(content=content, script=script, category="mixed", difficulty=difficulty)
    
    def generate_technical_probe(self, script: str = "devanagari", difficulty: str = "medium") -> IndicProbe:
        """Generate a technical document style probe with numbers and mixed content."""
        script_data = self.SCRIPTS.get(script, self.DEVANAGARI)
        
        # Create a technical-looking document structure
        lines = []
        
        # Header-like content
        lines.append(self._generate_word(script_data, 3) + " " + self.rng.choice(self.TECH_TERMS))
        lines.append("")
        
        # Numbered list items
        for i in range(3):
            num = self._generate_number(script_data, 1)
            word = self._generate_word(script_data, 2)
            tech = self.rng.choice(self.TECH_TERMS)
            lines.append(f"{num}. {word} ({tech})")
        
        content = "\n".join(lines)
        
        return IndicProbe(content=content, script=script, category="technical", difficulty=difficulty)
    
    def generate_batch(
        self,
        count: int = 100,
        scripts: List[str] = None,
        difficulties: List[str] = None
    ) -> List[IndicProbe]:
        """Generate a batch of Indic probes."""
        scripts = scripts or ["devanagari", "tamil", "telugu", "kannada", "malayalam"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        categories = ["pure", "mixed", "technical"]
        
        probes = []
        for _ in range(count):
            script = self.rng.choice(scripts)
            difficulty = self.rng.choice(difficulties)
            category = self.rng.choice(categories)
            
            if category == "pure":
                probe = self.generate_pure_probe(script, difficulty)
            elif category == "mixed":
                probe = self.generate_mixed_probe(script, difficulty)
            else:
                probe = self.generate_technical_probe(script, difficulty)
            
            probes.append(probe)
        
        return probes
    
    def get_corpus(self, count: int = 500) -> str:
        """Get probes as a single corpus string."""
        probes = self.generate_batch(count)
        return "\n\n".join(p.content for p in probes)
    
    def get_per_script_stats(self, probes: List[IndicProbe]) -> Dict[str, Dict[str, int]]:
        """Get statistics per script."""
        stats = {}
        for probe in probes:
            if probe.script not in stats:
                stats[probe.script] = {"pure": 0, "mixed": 0, "technical": 0, "total_chars": 0}
            stats[probe.script][probe.category] += 1
            stats[probe.script]["total_chars"] += len(probe.content)
        return stats
