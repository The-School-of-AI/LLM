"""
MCQ Probe Generator - Multiple choice question format templates.

Generates MCQ FORMAT structures for tokenizer evaluation WITHOUT
using real benchmark content. Tests tokenizer handling of:
- Option markers (A/B/C/D, 1/2/3/4, bullets)
- Question structure formatting
- Answer choice patterns
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class MCQProbe:
    """A single MCQ format probe."""
    content: str
    style: str  # abcd, numbered, bullet
    difficulty: str


class MCQProbeGenerator:
    """
    Generates MCQ format-only probes.
    
    Creates question STRUCTURES without semantic content to test
    tokenizer behavior on MCQ formatting.
    """
    
    # Placeholder question stems
    QUESTION_STEMS = [
        "Which of the following {topic} is {property}?",
        "Select the {property} {topic}:",
        "What is the {property} of {topic}?",
        "Identify the correct {topic}:",
        "Which option represents {topic}?",
        "Choose the {property} answer:",
        "The {topic} that {action} is:",
        "According to {source}, which {topic}?",
    ]
    
    # Placeholder topics (generic, non-benchmark)
    TOPICS = [
        "element", "option", "value", "method", "approach",
        "statement", "expression", "result", "factor", "term",
    ]
    
    # Placeholder properties (generic)
    PROPERTIES = [
        "correct", "valid", "appropriate", "suitable", "accurate",
        "primary", "optimal", "equivalent", "true", "best",
    ]
    
    # Option content placeholders
    OPTION_PLACEHOLDERS = [
        "[OPTION_{letter}]",
        "Option {letter} content here",
        "Choice {letter}: {placeholder}",
        "{placeholder_word}_{number}",
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _generate_options_abcd(self, num_options: int = 4) -> List[str]:
        """Generate A/B/C/D style options."""
        letters = 'ABCDEFGH'[:num_options]
        options = []
        for letter in letters:
            placeholder = self.rng.choice(self.OPTION_PLACEHOLDERS).format(
                letter=letter,
                placeholder=f"content_{self.rng.randint(1, 100)}",
                placeholder_word=self.rng.choice(["alpha", "beta", "gamma", "delta", "epsilon"]),
                number=self.rng.randint(1, 999),
            )
            options.append(f"{letter}) {placeholder}")
        return options
    
    def _generate_options_numbered(self, num_options: int = 4) -> List[str]:
        """Generate 1/2/3/4 style options."""
        options = []
        for i in range(1, num_options + 1):
            placeholder = self.rng.choice(self.OPTION_PLACEHOLDERS).format(
                letter=str(i),
                placeholder=f"item_{self.rng.randint(1, 100)}",
                placeholder_word=self.rng.choice(["first", "second", "third", "fourth", "fifth"]),
                number=self.rng.randint(1, 999),
            )
            options.append(f"{i}. {placeholder}")
        return options
    
    def _generate_options_bullet(self, num_options: int = 4) -> List[str]:
        """Generate bullet style options."""
        bullets = ["•", "-", "*", "→"][: num_options]
        options = []
        for i, bullet in enumerate(bullets):
            placeholder = self.rng.choice(self.OPTION_PLACEHOLDERS).format(
                letter=chr(65 + i),
                placeholder=f"choice_{self.rng.randint(1, 100)}",
                placeholder_word=self.rng.choice(["option", "selection", "answer", "choice"]),
                number=self.rng.randint(1, 999),
            )
            options.append(f"{bullet} {placeholder}")
        return options
    
    def _generate_question_stem(self) -> str:
        """Generate a placeholder question stem."""
        template = self.rng.choice(self.QUESTION_STEMS)
        return template.format(
            topic=self.rng.choice(self.TOPICS),
            property=self.rng.choice(self.PROPERTIES),
            action="satisfies the condition",
            source="the given context",
        )
    
    def generate_probe(
        self,
        style: str = "abcd",
        difficulty: str = "medium",
        num_options: int = 4
    ) -> MCQProbe:
        """
        Generate a single MCQ format probe.
        
        Args:
            style: One of 'abcd', 'numbered', 'bullet'
            difficulty: One of 'easy', 'medium', 'hard'
            num_options: Number of answer choices
        """
        question = self._generate_question_stem()
        
        if style == "abcd":
            options = self._generate_options_abcd(num_options)
        elif style == "numbered":
            options = self._generate_options_numbered(num_options)
        else:
            options = self._generate_options_bullet(min(num_options, 4))
        
        # Format based on difficulty (affects whitespace/structure)
        if difficulty == "easy":
            # Simple format
            content = f"Question: {question}\n" + "\n".join(options)
        elif difficulty == "medium":
            # Standard format with answer prompt
            content = f"Q: {question}\n\n" + "\n".join(options) + "\n\nAnswer: ___"
        else:
            # Complex format with metadata
            content = f"""Question #{self.rng.randint(1, 100)}:
{question}

{chr(10).join(options)}

Select your answer: [ ]
Confidence: ____%"""
        
        return MCQProbe(content=content, style=style, difficulty=difficulty)
    
    def generate_batch(
        self,
        count: int = 100,
        styles: List[str] = None,
        difficulties: List[str] = None
    ) -> List[MCQProbe]:
        """Generate a batch of MCQ probes."""
        styles = styles or ["abcd", "numbered", "bullet"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        
        probes = []
        for _ in range(count):
            style = self.rng.choice(styles)
            difficulty = self.rng.choice(difficulties)
            num_options = self.rng.choice([3, 4, 5])
            probe = self.generate_probe(style, difficulty, num_options)
            probes.append(probe)
        
        return probes
    
    def get_corpus(self, count: int = 500) -> str:
        """Get probes as a single corpus string."""
        probes = self.generate_batch(count)
        return "\n\n---\n\n".join(p.content for p in probes)
