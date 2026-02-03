"""
Synthetic Instruction Generator - Instruction format templates.

Generates synthetic instruction patterns for tokenizer evaluation
WITHOUT using real benchmark prompts. Tests tokenizer handling of:
- Instruction formatting
- Multi-step reasoning patterns
- Diverse prompt styles
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class InstructionProbe:
    """A single synthetic instruction probe."""
    content: str
    style: str  # direct, step_by_step, conversational, formal
    difficulty: str


class SyntheticInstructionGenerator:
    """
    Generates synthetic instruction probes.
    
    Creates instruction FORMAT templates without real benchmark
    content to test tokenizer behavior on prompt structures.
    """
    
    # Direct instruction templates
    DIRECT_TEMPLATES = [
        "{action} the following {object}: {placeholder}",
        "Please {action} this {object}.",
        "{action} {object} and return the result.",
        "Your task is to {action} the given {object}.",
        "Given {object}, {action} it accordingly.",
    ]
    
    # Step-by-step templates
    STEP_BY_STEP_TEMPLATES = [
        """Follow these steps:
1. First, {step1}
2. Then, {step2}
3. Finally, {step3}""",
        
        """To complete this task:
- Step 1: {step1}
- Step 2: {step2}
- Step 3: {step3}
- Step 4: {step4}""",
        
        """Instructions:
a) {step1}
b) {step2}
c) {step3}

Output the final result.""",
    ]
    
    # Conversational templates
    CONVERSATIONAL_TEMPLATES = [
        "Hey, can you help me {action} this {object}?",
        "I need to {action} something. Here's the {object}: {placeholder}",
        "Would you mind {action_ing} the following for me?\n\n{placeholder}",
        "Quick question - how would you {action} this {object}?",
    ]
    
    # Formal templates
    FORMAL_TEMPLATES = [
        """## Task Description

**Objective**: {action} the provided {object}.

**Input**: {placeholder}

**Expected Output**: {output_description}""",
        
        """### Instructions

You are required to {action} the following {object}. 
Ensure that your response follows these guidelines:
- {guideline1}
- {guideline2}
- {guideline3}

**Input Data**:
{placeholder}""",
        
        """TASK: {action}
INPUT TYPE: {object}
REQUIREMENTS: {requirements}

{placeholder}

Provide your response below:""",
    ]
    
    # System prompt templates
    SYSTEM_PROMPT_TEMPLATES = [
        "You are a helpful assistant that {capability}.",
        "Act as a {role} with expertise in {domain}.",
        "You are an AI designed to {task}. Always {constraint}.",
        "Your role is to {role_description}. Remember to {reminder}.",
    ]
    
    # Placeholder components
    ACTIONS = [
        "analyze", "summarize", "explain", "transform", "evaluate",
        "process", "review", "compare", "classify", "extract",
        "describe", "validate", "convert", "optimize", "debug",
    ]
    
    OBJECTS = [
        "data", "content", "input", "information", "text",
        "structure", "format", "pattern", "sequence", "collection",
    ]
    
    STEPS = [
        "read the input carefully",
        "identify key components",
        "apply the transformation",
        "verify the output",
        "format the result",
        "check for errors",
        "validate the structure",
        "optimize if needed",
    ]
    
    PLACEHOLDERS = [
        "[INPUT_DATA_HERE]",
        "{content_placeholder}",
        "<<SAMPLE_INPUT>>",
        "[PLACEHOLDER_CONTENT]",
        "___INPUT___",
    ]
    
    GUIDELINES = [
        "maintain consistency",
        "follow best practices",
        "ensure accuracy",
        "be concise",
        "include examples",
    ]
    
    ROLES = [
        "expert analyst", "technical reviewer", "helpful assistant",
        "code specialist", "data scientist", "language expert",
    ]
    
    DOMAINS = [
        "text processing", "data analysis", "code review",
        "information extraction", "pattern recognition",
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _action_to_ing(self, action: str) -> str:
        """Convert action to -ing form."""
        if action.endswith('e'):
            return action[:-1] + 'ing'
        return action + 'ing'
    
    def generate_direct_probe(self, difficulty: str = "medium") -> InstructionProbe:
        """Generate a direct instruction probe."""
        template = self.rng.choice(self.DIRECT_TEMPLATES)
        
        content = template.format(
            action=self.rng.choice(self.ACTIONS),
            object=self.rng.choice(self.OBJECTS),
            placeholder=self.rng.choice(self.PLACEHOLDERS),
        )
        
        return InstructionProbe(content=content, style="direct", difficulty=difficulty)
    
    def generate_step_by_step_probe(self, difficulty: str = "medium") -> InstructionProbe:
        """Generate a step-by-step instruction probe."""
        template = self.rng.choice(self.STEP_BY_STEP_TEMPLATES)
        
        steps = self.rng.sample(self.STEPS, min(4, len(self.STEPS)))
        
        content = template.format(
            step1=steps[0] if len(steps) > 0 else "begin",
            step2=steps[1] if len(steps) > 1 else "continue",
            step3=steps[2] if len(steps) > 2 else "finish",
            step4=steps[3] if len(steps) > 3 else "verify",
        )
        
        return InstructionProbe(content=content, style="step_by_step", difficulty=difficulty)
    
    def generate_conversational_probe(self, difficulty: str = "medium") -> InstructionProbe:
        """Generate a conversational instruction probe."""
        template = self.rng.choice(self.CONVERSATIONAL_TEMPLATES)
        action = self.rng.choice(self.ACTIONS)
        
        content = template.format(
            action=action,
            action_ing=self._action_to_ing(action),
            object=self.rng.choice(self.OBJECTS),
            placeholder=self.rng.choice(self.PLACEHOLDERS),
        )
        
        return InstructionProbe(content=content, style="conversational", difficulty=difficulty)
    
    def generate_formal_probe(self, difficulty: str = "medium") -> InstructionProbe:
        """Generate a formal instruction probe."""
        template = self.rng.choice(self.FORMAL_TEMPLATES)
        guidelines = self.rng.sample(self.GUIDELINES, 3)
        
        content = template.format(
            action=self.rng.choice(self.ACTIONS),
            object=self.rng.choice(self.OBJECTS),
            placeholder=self.rng.choice(self.PLACEHOLDERS),
            output_description=f"A {self.rng.choice(self.OBJECTS)} in processed form",
            guideline1=guidelines[0],
            guideline2=guidelines[1],
            guideline3=guidelines[2],
            requirements=", ".join(self.rng.sample(self.GUIDELINES, 2)),
        )
        
        return InstructionProbe(content=content, style="formal", difficulty=difficulty)
    
    def generate_system_prompt_probe(self, difficulty: str = "medium") -> InstructionProbe:
        """Generate a system prompt style probe."""
        template = self.rng.choice(self.SYSTEM_PROMPT_TEMPLATES)
        
        content = template.format(
            capability=f"{self.rng.choice(self.ACTIONS)}s {self.rng.choice(self.OBJECTS)}",
            role=self.rng.choice(self.ROLES),
            domain=self.rng.choice(self.DOMAINS),
            task=f"{self.rng.choice(self.ACTIONS)} {self.rng.choice(self.OBJECTS)}",
            constraint=self.rng.choice(self.GUIDELINES),
            role_description=f"{self.rng.choice(self.ACTIONS)} and {self.rng.choice(self.ACTIONS)}",
            reminder=self.rng.choice(self.GUIDELINES),
        )
        
        return InstructionProbe(content=content, style="system", difficulty=difficulty)
    
    def generate_batch(
        self,
        count: int = 100,
        styles: List[str] = None,
        difficulties: List[str] = None
    ) -> List[InstructionProbe]:
        """Generate a batch of instruction probes."""
        styles = styles or ["direct", "step_by_step", "conversational", "formal", "system"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        
        generators = {
            "direct": self.generate_direct_probe,
            "step_by_step": self.generate_step_by_step_probe,
            "conversational": self.generate_conversational_probe,
            "formal": self.generate_formal_probe,
            "system": self.generate_system_prompt_probe,
        }
        
        probes = []
        for _ in range(count):
            style = self.rng.choice(styles)
            difficulty = self.rng.choice(difficulties)
            probe = generators[style](difficulty)
            probes.append(probe)
        
        return probes
    
    def get_corpus(self, count: int = 500) -> str:
        """Get probes as a single corpus string."""
        probes = self.generate_batch(count)
        return "\n\n---\n\n".join(p.content for p in probes)
