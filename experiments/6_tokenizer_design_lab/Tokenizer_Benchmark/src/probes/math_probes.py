"""
Math Probe Generator - Format-only mathematical content.

Generates synthetic math content for tokenizer evaluation WITHOUT
using real benchmark questions. Focuses on:
- LaTeX equation structures
- Numeric patterns
- Mathematical notation
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class MathProbe:
    """A single math probe for tokenizer evaluation."""
    content: str
    category: str  # latex, numeric, expression, etc.
    difficulty: str  # easy, medium, hard


class MathProbeGenerator:
    """
    Generates format-only math probes.
    
    These are synthetic mathematical expressions that test tokenizer
    behavior on math formatting WITHOUT containing real benchmark content.
    """
    
    # LaTeX templates - structure only, no semantic content
    LATEX_TEMPLATES = [
        r"$\frac{{{a}}}{{{b}}} + {c} = {d}$",
        r"$\sqrt{{{a}}} \times {b} = {c}$",
        r"$\sum_{{i=1}}^{{{n}}} {expr}$",
        r"$\int_{{{a}}}^{{{b}}} {expr} \, dx$",
        r"$\lim_{{x \to {a}}} {expr}$",
        r"$\binom{{{n}}}{{{k}}}$",
        r"${a}^{{{b}}} \cdot {c}^{{{d}}}$",
        r"$\log_{{{base}}}({arg})$",
        r"$\sin({a}) + \cos({b}) = {c}$",
        r"$\frac{{d}}{{dx}} ({expr})$",
        r"$\prod_{{i={a}}}^{{{b}}} {expr}$",
        r"$\begin{{matrix}} {a} & {b} \\ {c} & {d} \end{{matrix}}$",
        r"$|{a} - {b}| \leq {c}$",
        r"${a} \equiv {b} \pmod{{{m}}}$",
        r"$\vec{{{v}}} \cdot \vec{{{w}}} = {result}$",
    ]
    
    # Numeric pattern templates
    NUMERIC_PATTERNS = [
        "{a}, {b}, {c}, {d}, ...",  # Arithmetic sequence
        "{a}, {b}, {c}, {d}, {e}",  # General sequence
        "{a}/{b} + {c}/{d} = ?",     # Fraction addition
        "{a} × {b} + {c} = ?",       # BODMAS
        "{a}² + {b}² = ?",           # Squares
        "√{a} × √{b} = ?",           # Square roots
        "{a}! / {b}! = ?",           # Factorials
        "GCD({a}, {b}) = ?",         # GCD
        "LCM({a}, {b}) = ?",         # LCM
        "{a} mod {b} = ?",           # Modulo
    ]
    
    # Expression templates
    EXPRESSION_TEMPLATES = [
        "Let x = {a}. Then x + {b} = ?",
        "If y = {a}x + {b}, find y when x = {c}",
        "Solve: {a}x + {b} = {c}",
        "Simplify: ({a} + {b})({c} - {d})",
        "Expand: ({a}x + {b})²",
        "Factor: x² + {sum}x + {prod}",
        "Calculate: {a}% of {b}",
        "Convert {a}/{b} to decimal",
        "Express {a}:{b} as a fraction",
        "Find {a}th term of AP: {first}, {second}, ...",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.rng = random.Random(seed)
    
    def _random_int(self, low: int = 1, high: int = 100) -> int:
        return self.rng.randint(low, high)
    
    def _random_var(self) -> str:
        return self.rng.choice(['x', 'y', 'z', 'n', 'k', 'm', 'a', 'b'])
    
    def generate_latex_probe(self, difficulty: str = "medium") -> MathProbe:
        """Generate a LaTeX format probe."""
        template = self.rng.choice(self.LATEX_TEMPLATES)
        
        # Difficulty affects number complexity
        if difficulty == "easy":
            max_val = 10
        elif difficulty == "medium":
            max_val = 50
        else:
            max_val = 500
        
        # Fill in template
        content = template.format(
            a=self._random_int(1, max_val),
            b=self._random_int(1, max_val),
            c=self._random_int(1, max_val),
            d=self._random_int(1, max_val),
            e=self._random_int(1, max_val),
            n=self._random_int(1, 20),
            k=self._random_int(1, 10),
            m=self._random_int(2, 13),
            base=self._random_int(2, 10),
            arg=self._random_int(1, max_val),
            expr=f"{self._random_var()}^2 + {self._random_int(1, 10)}",
            v=self._random_var(),
            w=self.rng.choice(['u', 'v', 'w']),
            result=self._random_int(1, max_val),
        )
        
        return MathProbe(content=content, category="latex", difficulty=difficulty)
    
    def generate_numeric_probe(self, difficulty: str = "medium") -> MathProbe:
        """Generate a numeric pattern probe."""
        template = self.rng.choice(self.NUMERIC_PATTERNS)
        
        if difficulty == "easy":
            max_val = 20
        elif difficulty == "medium":
            max_val = 100
        else:
            max_val = 1000
        
        # For arithmetic sequences, generate actual sequences
        if "..." in template:
            start = self._random_int(1, max_val // 4)
            diff = self._random_int(1, 10)
            content = template.format(
                a=start,
                b=start + diff,
                c=start + 2*diff,
                d=start + 3*diff,
                e=start + 4*diff,
            )
        else:
            content = template.format(
                a=self._random_int(1, max_val),
                b=self._random_int(1, max_val),
                c=self._random_int(1, max_val),
                d=self._random_int(1, max_val),
                e=self._random_int(1, max_val),
            )
        
        return MathProbe(content=content, category="numeric", difficulty=difficulty)
    
    def generate_expression_probe(self, difficulty: str = "medium") -> MathProbe:
        """Generate an algebraic expression probe."""
        template = self.rng.choice(self.EXPRESSION_TEMPLATES)
        
        if difficulty == "easy":
            max_val = 10
        elif difficulty == "medium":
            max_val = 50
        else:
            max_val = 200
        
        # For factoring, create valid sum/product
        p = self._random_int(1, 15)
        q = self._random_int(1, 15)
        
        content = template.format(
            a=self._random_int(1, max_val),
            b=self._random_int(1, max_val),
            c=self._random_int(1, max_val),
            d=self._random_int(1, max_val),
            sum=p + q,
            prod=p * q,
            first=self._random_int(1, max_val),
            second=self._random_int(1, max_val) + self._random_int(1, 10),
        )
        
        return MathProbe(content=content, category="expression", difficulty=difficulty)
    
    def generate_batch(
        self,
        count: int = 100,
        categories: List[str] = None,
        difficulties: List[str] = None
    ) -> List[MathProbe]:
        """
        Generate a batch of math probes.
        
        Args:
            count: Number of probes to generate
            categories: List of categories (latex, numeric, expression)
            difficulties: List of difficulties (easy, medium, hard)
        
        Returns:
            List of MathProbe objects
        """
        categories = categories or ["latex", "numeric", "expression"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        
        generators = {
            "latex": self.generate_latex_probe,
            "numeric": self.generate_numeric_probe,
            "expression": self.generate_expression_probe,
        }
        
        probes = []
        for _ in range(count):
            category = self.rng.choice(categories)
            difficulty = self.rng.choice(difficulties)
            probe = generators[category](difficulty)
            probes.append(probe)
        
        return probes
    
    def get_corpus(self, count: int = 500) -> str:
        """Get probes as a single corpus string."""
        probes = self.generate_batch(count)
        return "\n".join(p.content for p in probes)
