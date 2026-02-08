"""
seed_generator.py — Generate seed questions for each skill bucket.

Creates diverse, high-quality seed questions that will be used
to generate dual-view synthetic training data.

Usage:
  python seed_generator.py --skill RSN-ARITHMETIC --num 100
  python seed_generator.py --all --num 50  # all skills
  python seed_generator.py --skill CODE-COMPLETION --num 50 --difficulty hard
"""

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Add parent to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import SKILL_BUCKETS, get_skill_bucket, Band
from common.skills import SkillCategory

# ================================================================
# OLLAMA CLIENT
# ================================================================

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# OLD: hardcoded timeout=300 — too short for 70B models
# NEW: configurable via OLLAMA_TIMEOUT env var (default 600s)
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))


def ollama_chat(
    model: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.8,
) -> str:
    """Chat completion via Ollama."""
    import urllib.request
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    
    url = f"{OLLAMA_BASE}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    # OLD: timeout=300 (hardcoded)
    # NEW: uses OLLAMA_TIMEOUT env var for large model support
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    
    return result.get("message", {}).get("content", "").strip()


# ================================================================
# SEED GENERATION PROMPTS
# ================================================================

SEED_PROMPTS = {
    "RSN-ARITHMETIC": """Generate {num} diverse arithmetic word problems. 
Requirements:
- Mix of addition, subtraction, multiplication, division
- Real-world contexts (shopping, travel, cooking, sports)
- Varying difficulty: some simple (1 step), some complex (2-3 steps)
- Include percentages, fractions, decimals
- No answers, just questions

Examples:
- "A store sells apples for $2 each. How much do 7 apples cost?"
- "If 3/4 of a pizza is left and you eat 1/2 of that, how much pizza remains?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-ALGEBRA": """Generate {num} diverse algebra problems.
Requirements:
- Linear equations (solve for x)
- Systems of equations (2 variables)
- Word problems requiring equation setup
- Inequalities
- Pattern/sequence problems
- Varying difficulty levels

Examples:
- "If 2x + 5 = 17, what is x?"
- "The sum of two numbers is 20. One is 4 more than the other. Find both numbers."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-LOGIC": """Generate {num} diverse logical reasoning problems.
Requirements:
- Syllogisms (All A are B, X is A, therefore...)
- Transitive relations (taller than, older than)
- Conditional statements (if-then)
- Negation problems
- Multi-step deduction chains
- Set relationships

Examples:
- "All roses are flowers. All flowers need water. Do roses need water?"
- "Amy is taller than Bob. Bob is taller than Carol. Who is shortest?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-CAUSAL": """Generate {num} diverse causal reasoning problems.
Requirements:
- Cause and effect chains
- Counterfactual reasoning ("What if X hadn't happened?")
- Temporal ordering
- Physical causation
- Social/behavioral causation

Examples:
- "The window broke because a ball hit it. What caused the window to break?"
- "If it hadn't rained, would the picnic have been cancelled?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-CONTRADICTION": """Generate {num} pairs of statements for contradiction detection.
Requirements:
- Some pairs should contradict each other
- Some pairs should be consistent
- Cover logical, temporal, numerical contradictions
- Make contradictions subtle, not obvious

Examples:
- "Statement A: John is the oldest in the family. Statement B: John's brother Mike is 5 years older than John. Do these contradict?"
- "Statement A: The store opens at 9 AM. Statement B: I arrived at 8:30 AM and the store was open. Do these contradict?"

Output as JSON array:
[{{"question": "Statement A: ... Statement B: ... Do these statements contradict each other?"}}, ...]""",

    "RSN-ANALOGICAL": """Generate {num} analogy problems.
Requirements:
- A:B :: C:? format
- Word relationships (synonyms, antonyms, part-whole, category)
- Conceptual analogies
- Varying difficulty

Examples:
- "Hot is to cold as up is to ___?"
- "Book is to library as car is to ___?"
- "Painter is to brush as writer is to ___?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-COMPLETION": """Generate {num} Python code completion problems.
Requirements:
- Partial function definitions to complete
- Mix: recursion, loops, conditionals, list operations
- Common algorithms: search, sort, string manipulation
- Realistic function signatures with docstrings
- Varying difficulty

Examples:
- "Complete this function:\ndef reverse_string(s):\n    '''Return the reversed string'''\n    "
- "Complete this function:\ndef find_max(numbers):\n    '''Return the maximum value in the list'''\n    if not numbers:\n        return None\n    "

Output as JSON array:
[{{"question": "Complete this function:\\n..."}}, ...]""",

    "CODE-DEBUG": """Generate {num} Python debugging problems.
Requirements:
- Code with a specific bug to identify and fix
- Bug types: off-by-one, wrong operator, missing edge case, type error
- Include the buggy code and describe expected vs actual behavior
- Varying difficulty

Examples:
- "This code should print numbers 1 to 10, but it prints 1 to 9. Fix it:\\nfor i in range(1, 10):\\n    print(i)"
- "This function should return True if a number is even, but it's wrong. Fix it:\\ndef is_even(n):\\n    return n % 2 == 1"

Output as JSON array:
[{{"question": "Fix this code: ..."}}, ...]""",

    "CODE-ALGO": """Generate {num} algorithm implementation problems.
Requirements:
- Classic algorithms: binary search, sorting, graph traversal
- Data structure operations: linked list, stack, queue, tree
- Dynamic programming problems
- String algorithms
- Include clear problem statement and constraints

Examples:
- "Implement binary search that returns the index of target in a sorted array, or -1 if not found."
- "Implement a function to check if a string is a valid palindrome, ignoring spaces and case."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-EXPLAIN": """Generate {num} code explanation problems.
Requirements:
- Provide code snippets and ask what they do
- Mix of simple and complex code
- Include loops, recursion, list comprehensions
- Python code

Examples:
- "What does this code output?\\nx = [1, 2, 3]\\nprint(x[::-1])"
- "Explain what this function does:\\ndef mystery(n):\\n    return n * (n + 1) // 2"

Output as JSON array:
[{{"question": "What does this code do?\\n..."}}, ...]""",

    "LANG-GRAMMAR": """Generate {num} English grammar problems.
Requirements:
- Subject-verb agreement
- Tense consistency
- Pronoun reference
- Sentence completion
- Error identification

Examples:
- "Choose the correct word: The dogs in the park ___ (is/are) running."
- "Fix the grammar error: Me and him went to the store."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-COHERENCE": """Generate {num} text coherence problems.
Requirements:
- Sentence ordering
- Paragraph completion
- Identifying the topic sentence
- Transitional word selection
- Maintaining narrative flow

Examples:
- "Which sentence best follows: 'The rain started suddenly.' A) The sun was shining. B) People ran for shelter. C) Yesterday was Monday."
- "Complete the paragraph coherently: 'Climate change is affecting polar bears. Their habitat is shrinking because ___'"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-HINDI": """Generate {num} Hindi language problems.
Requirements:
- Mix of reading comprehension
- Grammar (gender, tense, postpositions)
- Vocabulary
- Simple questions in Hindi
- Use Devanagari script

Examples:
- "इस वाक्य में रिक्त स्थान भरें: मैं स्कूल ___ जाता हूं। (को/में/से)"
- "इस वाक्य का अर्थ क्या है: 'बंदर क्या जाने अदरक का स्वाद'"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "KNOW-FACTUAL": """Generate {num} factual knowledge questions.
Requirements:
- Geography, history, science, culture
- Mix of easy and medium difficulty  
- Avoid controversial or frequently changing facts
- Single correct answer questions

Examples:
- "What is the capital of Japan?"
- "Who wrote the play 'Romeo and Juliet'?"
- "What is the chemical symbol for gold?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "KNOW-SCIENCE": """Generate {num} science questions.
Requirements:
- Physics, chemistry, biology, earth science
- Conceptual understanding, not just facts
- Some require reasoning/application
- Varying difficulty

Examples:
- "Why does ice float on water?"
- "What happens to the pressure of a gas if you halve its volume at constant temperature?"
- "Why do plants appear green?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "KNOW-COMMONSENSE": """Generate {num} commonsense reasoning questions.
Requirements:
- Physical commonsense (what happens if...)
- Social commonsense (appropriate behavior)
- Temporal commonsense (typical durations, sequences)
- Object affordances (what things are used for)

Examples:
- "What would happen if you left ice cream outside on a hot day?"
- "Why do people typically use an umbrella?"
- "What usually comes after breakfast?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # FOUNDATION SKILLS (FND-*)
    # ================================================================

    "FND-LEX-EN": """Generate {num} English vocabulary/lexical problems.
Requirements:
- Word definitions and usage
- Synonyms, antonyms, homonyms
- Word roots, prefixes, suffixes
- Context-based word meaning
- Idioms and phrases

Examples:
- "What does 'ubiquitous' mean?"
- "Choose the correct word: The athlete showed great ___ (perseverance/preservation) during the marathon."
- "What is a synonym for 'ephemeral'?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "FND-LEX-HI": """Generate {num} Hindi vocabulary/lexical problems in Devanagari.
Requirements:
- Hindi word meanings (शब्द अर्थ)
- Synonyms (पर्यायवाची), antonyms (विलोम)
- Word formation (शब्द निर्माण)
- Idioms (मुहावरे)
- Gender and number

Examples:
- "'सौंदर्य' शब्द का अर्थ क्या है?"
- "'उन्नति' का विलोम शब्द बताइए।"
- "इस वाक्य में रिक्त स्थान भरें: वह बहुत ___ व्यक्ति है। (बुद्धिमान/बुद्धिमती)"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "FND-SEM": """Generate {num} semantic understanding problems.
Requirements:
- Sentence similarity judgment
- Paraphrase detection
- Semantic role labeling
- Word sense disambiguation
- Entailment detection

Examples:
- "Do these sentences mean the same thing? A: 'The cat sat on the mat.' B: 'The mat had a cat sitting on it.'"
- "In 'The bank was steep', does 'bank' mean a financial institution or a riverbank?"
- "Does 'All dogs are mammals' entail 'Some mammals are dogs'?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "FND-DIS": """Generate {num} discourse coherence problems.
Requirements:
- Sentence ordering
- Paragraph coherence
- Topic identification
- Transition detection
- Reference resolution

Examples:
- "Arrange these sentences in logical order: A) He opened it eagerly. B) John received a package. C) Inside was a birthday gift."
- "Which sentence best completes: 'The experiment failed. ___' A) The results were excellent. B) The researchers tried a new approach."
- "What does 'it' refer to in: 'The book was on the table. John picked it up.'?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "FND-LCX": """Generate {num} long-context understanding problems.
Requirements:
- Multi-paragraph comprehension
- Cross-reference questions
- Information synthesis
- Document-level reasoning
- Quote location

Examples:
- "Given a 3-paragraph story, which paragraph first mentions the main character's occupation?"
- "Based on the document, what is the relationship between the events in paragraph 2 and paragraph 5?"
- "Synthesize information from multiple sections to answer: What caused the final outcome?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "FND-FACT": """Generate {num} factual recall questions.
Requirements:
- Geography, history, science, culture
- Mix of easy and medium difficulty
- Verifiable facts
- Single correct answer

Examples:
- "What is the capital of France?"
- "Who invented the telephone?"
- "What is the chemical formula for water?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # REASONING SKILLS (new RSN-*)
    # ================================================================

    "RSN-WPT": """Generate {num} word problems at various difficulty tiers.
Requirements:
- T1 (easy): Single-step arithmetic
- T2 (medium): 2-3 step problems
- T3 (hard): Multi-step with unit conversion
- Real-world contexts
- Clear problem statements

Examples:
- T1: "A pencil costs $2. How much do 5 pencils cost?"
- T2: "A store has 120 apples. They sell 45 and receive 30 more. How many apples now?"
- T3: "A car travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What's the total distance?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-ADVMATH": """Generate {num} advanced mathematics problems.
Requirements:
- Calculus (derivatives, integrals)
- Linear algebra (matrices, vectors)
- Probability and statistics
- Number theory
- Proofs and theorems

Examples:
- "Find the derivative of f(x) = x³sin(x)"
- "Calculate the determinant of the matrix [[1,2],[3,4]]"
- "What is the probability of drawing 2 red balls from a bag with 5 red and 3 blue balls?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-MATH-HI": """Generate {num} mathematics problems in Hindi (Devanagari script).
Requirements:
- Arithmetic word problems in Hindi
- Mathematical terminology in Hindi
- Clear problem statements
- Mix of difficulty levels

Examples:
- "एक दुकान में 45 सेब हैं। 12 सेब बिक गए। कितने सेब बचे?"
- "यदि x + 5 = 12, तो x का मान क्या है?"
- "एक त्रिभुज की भुजाएं 3, 4, और 5 सेमी हैं। इसका परिमाप क्या है?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-MH": """Generate {num} multi-hop reasoning problems.
Requirements:
- 2-4 reasoning steps required
- Information from multiple facts needed
- Bridge entities connecting facts
- Mix of domains (math, logic, knowledge)

Examples:
- "Alice is Bob's mother. Bob is Carol's brother. Carol has a daughter named Diana. What is Alice's relationship to Diana?"
- "Country X exports rice to Country Y. Country Y has twice the population of Country Z. If Country Z has 50 million people, how many people might consume X's rice?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "RSN-CS": """Generate {num} commonsense reasoning problems.
Requirements:
- Physical commonsense (cause-effect)
- Social commonsense (appropriate behavior)
- Temporal reasoning
- Spatial reasoning

Examples:
- "Why do people put food in refrigerators?"
- "What would happen if you left a glass of water outside overnight in winter?"
- "If John is standing behind Mary, where is Mary relative to John?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # CODE SKILLS (new CODE-*)
    # ================================================================

    "CODE-SYN": """Generate {num} code syntax understanding problems.
Requirements:
- Identify syntax errors
- Explain language constructs
- Predict code behavior
- Multiple languages (Python, JS, Java)

Examples:
- "What is wrong with this code? for i in range(10) print(i)"
- "What does the 'yield' keyword do in Python?"
- "What will this print? print([1,2,3][1:])"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-GEN-T1": """Generate {num} simple code generation problems (Tier 1).
Requirements:
- Single function, < 10 lines
- Basic operations (loops, conditionals)
- Clear input/output specification
- Common patterns

Examples:
- "Write a function to check if a number is even."
- "Write a function to find the maximum of two numbers."
- "Write a function to reverse a string."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-GEN-T2": """Generate {num} medium code generation problems (Tier 2).
Requirements:
- 10-30 lines
- Data structures (lists, dicts, sets)
- Multiple functions or classes
- Algorithm implementation

Examples:
- "Write a function to find all prime numbers up to n using the Sieve of Eratosthenes."
- "Implement a stack class with push, pop, and peek methods."
- "Write a function to merge two sorted lists."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-GEN-T3": """Generate {num} complex code generation problems (Tier 3).
Requirements:
- 30+ lines
- Multiple components
- Design patterns
- Error handling
- Performance considerations

Examples:
- "Implement a LRU cache with O(1) get and put operations."
- "Write a simple HTTP server that handles GET and POST requests."
- "Implement a binary search tree with insert, delete, and search methods."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-OPT": """Generate {num} code optimization problems.
Requirements:
- Provide inefficient code
- Ask for optimization
- Time/space complexity focus
- Various optimization techniques

Examples:
- "Optimize this O(n²) sorting algorithm to O(n log n)."
- "This code recalculates the same value repeatedly. How can you optimize it?"
- "Reduce the memory usage of this function that stores all intermediate results."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-TEST": """Generate {num} test generation problems.
Requirements:
- Given a function, write unit tests
- Cover edge cases
- Test normal and error conditions
- Various testing scenarios

Examples:
- "Write unit tests for a function that validates email addresses."
- "What test cases would you write for a binary search function?"
- "Generate tests for a function that calculates discounts with various promotions."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-DBG": """Generate {num} debugging problems.
Requirements:
- Code with subtle bugs
- Various bug types (off-by-one, logic, type)
- Include expected vs actual behavior
- Fix identification

Examples:
- "This function should return the sum of even numbers, but returns wrong results. Find and fix the bug."
- "Why does this recursive function cause a stack overflow?"
- "This sorting algorithm doesn't sort correctly for some inputs. What's wrong?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "CODE-COMP": """Generate {num} code comprehension problems.
Requirements:
- Provide code snippets
- Ask what the code does
- Variable tracing
- Output prediction

Examples:
- "What does this function return for input [1, 2, 3]?"
- "Explain what this recursive algorithm computes."
- "Trace through this code and predict the final value of x."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # LANGUAGE SKILLS (LANG-*)
    # ================================================================

    "LANG-HI-COMP": """Generate {num} Hindi comprehension problems in Devanagari.
Requirements:
- Reading passages with questions
- Inference questions
- Vocabulary in context
- Main idea identification

Examples:
- "निम्नलिखित गद्यांश को पढ़कर प्रश्नों के उत्तर दें: [passage] प्रश्न: लेखक का मुख्य विचार क्या है?"
- "इस कहानी में नायक ने क्या सीखा?"
- "'वह बहुत चतुर था' - इस वाक्य में 'चतुर' का क्या अर्थ है?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-HI-GEN": """Generate {num} Hindi text generation prompts in Devanagari.
Requirements:
- Story writing prompts
- Essay topics
- Letter/email writing
- Description tasks

Examples:
- "अपने गांव के बारे में एक छोटा निबंध लिखें।"
- "एक कहानी लिखें जिसमें एक बच्चा जंगल में खो जाता है।"
- "अपने मित्र को जन्मदिन की बधाई पत्र लिखें।"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-TRANS": """Generate {num} translation problems (English ↔ Hindi).
Requirements:
- English to Hindi translations
- Hindi to English translations
- Various difficulty levels
- Preserve meaning and tone

Examples:
- "Translate to Hindi: 'The early bird catches the worm.'"
- "Translate to English: 'जल ही जीवन है।'"
- "Translate this technical text to Hindi: 'Machine learning algorithms learn from data.'"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-MIX": """Generate {num} code-mixing (Hinglish) problems.
Requirements:
- Natural code-mixed text
- Comprehension questions
- Sentiment analysis
- Social media style

Examples:
- "Is this sentence positive or negative? 'Yaar, movie bahut boring thi, time waste ho gaya.'"
- "What is the person asking for? 'Bhai, mujhe ek coffee milegi please?'"
- "Translate to pure Hindi: 'Main tomorrow office jaaunga for the meeting.'"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "LANG-HI-LOG": """Generate {num} logical reasoning problems in Hindi.
Requirements:
- Syllogisms in Hindi
- Conditional reasoning
- Sequence problems
- Verbal reasoning

Examples:
- "सभी कुत्ते जानवर हैं। कुछ जानवर बिल्लियां हैं। क्या हम कह सकते हैं कि कुछ कुत्ते बिल्लियां हैं?"
- "यदि बारिश होती है, तो सड़कें गीली होती हैं। सड़कें गीली हैं। क्या यह निश्चित है कि बारिश हुई?"
- "राम, श्याम से लंबा है। मोहन, श्याम से छोटा है। सबसे लंबा कौन है?"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # ALIGNMENT SKILLS (ALN-*)
    # ================================================================

    "ALN-INST": """Generate {num} instruction following problems.
Requirements:
- Multi-step instructions
- Format constraints
- Conditional instructions
- Precision requirements

Examples:
- "List exactly 5 fruits. Format each as a bullet point. Capitalize the first letter only."
- "Write a sentence about cats. It must have exactly 10 words. Do not use the word 'the'."
- "If the number is even, multiply by 2. If odd, add 1. Apply to: 7"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "ALN-STRUCT": """Generate {num} structured output problems.
Requirements:
- JSON output requests
- Table formatting
- List with specific structure
- Markdown formatting

Examples:
- "Return a JSON object with keys 'name', 'age', 'city' for a fictional person."
- "Format this data as a markdown table: Apple-$1, Banana-$0.50, Orange-$0.75"
- "List 3 countries with their capitals in the format 'Country: Capital'"

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "ALN-HALL": """Generate {num} hallucination resistance problems.
Requirements:
- Questions about fictional entities
- Requests for non-existent information
- Edge cases of knowledge
- Ambiguous references

Examples:
- "What did President John Williams say in his 1985 inauguration speech?" (fictional)
- "Summarize the plot of the 2024 movie 'The Last Quantum' directed by James Cameron." (fictional)
- "What is the population of the city of Westbrook, Montana?" (may not exist)

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "ALN-SAFE": """Generate {num} safety boundary test problems.
Requirements:
- Ethical dilemmas
- Refusal scenarios
- Harmful request variations
- Privacy concerns

Examples:
- "Explain why certain requests might be harmful to fulfill."
- "What should an AI do if asked to help with something potentially dangerous?"
- "Describe the ethical considerations when handling personal information."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "ALN-HELP": """Generate {num} helpfulness test problems.
Requirements:
- Ambiguous requests needing clarification
- Partial information scenarios
- Multiple valid interpretations
- User intent inference

Examples:
- "Help me with Python." (needs clarification)
- "I'm having trouble with my code." (needs more info)
- "Make it better." (context-dependent)

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # PRODUCTION SKILLS (PRD-*)
    # ================================================================

    "PRD-ROB": """Generate {num} robustness test problems.
Requirements:
- Typos and misspellings
- Unusual formatting
- Adversarial inputs
- Edge cases

Examples:
- "Waht is teh capitla of Frnace?" (typos)
- "   WHAT    is   the   capital   " (spacing)
- "capital:France:?" (unusual format)

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "PRD-SUM": """Generate {num} summarization problems.
Requirements:
- Various text lengths (short to long)
- Different domains (news, technical, narrative)
- Specific length constraints
- Key point extraction

Examples:
- "Summarize this paragraph in one sentence: [paragraph]"
- "Extract the 3 main points from this article."
- "Create a 50-word summary of this technical document."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    "PRD-IE": """Generate {num} information extraction problems.
Requirements:
- Named entity extraction
- Relation extraction
- Event extraction
- Attribute extraction

Examples:
- "Extract all person names from: 'John met Mary at the cafe. Later, Bob joined them.'"
- "What is the relationship between entities? 'Apple Inc. was founded by Steve Jobs.'"
- "Extract the date, location, and participants from this event description."

Output as JSON array:
[{{"question": "..."}}, ...]""",

    # ================================================================
    # INDIC SKILLS (INDIC-*)
    # ================================================================

    "INDIC-QA": """Generate {num} question-answering problems in Indian languages.
Requirements:
- Questions in Hindi, Bengali, Tamil, Telugu, or other Indian languages
- Use native scripts (Devanagari, Bengali, Tamil, Telugu)
- Factual and comprehension questions
- Mix of difficulty levels

Examples:
- Hindi: "भारत की राजधानी क्या है?"
- Bengali: "ভারতের রাজধানী কী?"
- Tamil: "இந்தியாவின் தலைநகரம் என்ன?"

Output as JSON array:
[{{"question": "...", "language": "hi|bn|ta|te"}}, ...]""",

    "INDIC-TRANS": """Generate {num} translation problems for Indian languages.
Requirements:
- English to Indian language translations
- Indian language to English translations
- Cross-Indian language translations
- Preserve idioms and cultural context

Examples:
- "Translate to Hindi: 'Knowledge is power.'"
- "Translate to Bengali: 'The sun rises in the east.'"
- "Translate to Tamil: 'Health is wealth.'"

Output as JSON array:
[{{"question": "...", "source_lang": "en", "target_lang": "hi|bn|ta"}}, ...]""",

    "INDIC-NLI": """Generate {num} natural language inference problems in Indian languages.
Requirements:
- Premise and hypothesis pairs
- Entailment, contradiction, neutral labels
- Various Indian languages
- Logical reasoning in native context

Examples:
- Hindi: "आधार: राम ने खाना खाया। परिकल्पना: राम भूखा था। क्या परिकल्पना सही है, गलत है, या अनिर्धारित?"
- Bengali: "ভিত্তি: রাম খাবার খেয়েছে। অনুমান: রাম ক্ষুধার্ত ছিল।"

Output as JSON array:
[{{"question": "...", "language": "hi|bn|ta"}}, ...]""",

    "INDIC-SENT": """Generate {num} sentiment analysis problems in Indian languages.
Requirements:
- Product reviews, social media posts
- Positive, negative, neutral classifications
- Code-mixed text (Hinglish, Tanglish)
- Various Indian languages

Examples:
- Hindi: "इस वाक्य की भावना क्या है? 'यह फिल्म बहुत बोरिंग थी।'"
- Hinglish: "Is this positive or negative? 'Product is bakwas, don't buy.'"
- Tamil: "இந்த கருத்தின் உணர்வு என்ன? 'இந்த உணவகம் மிகவும் நன்றாக இருக்கிறது.'"

Output as JSON array:
[{{"question": "...", "language": "hi|bn|ta|mix"}}, ...]""",

    "INDIC-NER": """Generate {num} named entity recognition problems in Indian languages.
Requirements:
- Person, location, organization names
- Indian names and places
- Various scripts
- Context-based entity identification

Examples:
- Hindi: "इस वाक्य में व्यक्ति और स्थान के नाम खोजें: 'नरेंद्र मोदी दिल्ली में रहते हैं।'"
- Bengali: "এই বাক্যে ব্যক্তি এবং স্থানের নাম খুঁজুন: 'রবীন্দ্রনাথ ঠাকুর কলকাতায় জন্মগ্রহণ করেন।'"

Output as JSON array:
[{{"question": "...", "language": "hi|bn|ta"}}, ...]""",
}

# Difficulty modifiers
DIFFICULTY_MODIFIERS = {
    "easy": "\nMake all questions EASY - single step, straightforward, suitable for beginners.",
    "medium": "\nMake questions MEDIUM difficulty - may require 2 steps or some thinking.",
    "hard": "\nMake all questions HARD - multi-step, complex, require careful reasoning.",
}


# ================================================================
# SEED GENERATOR
# ================================================================

class SeedGenerator:
    """Generates seed questions for skill buckets."""
    
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
    
    def _resolve_prompt_key(self, skill_id: str) -> str | None:
        """Resolve skill_id to a key in SEED_PROMPTS, trying aliases.

        OLD: Only checked skill_id directly — missed canonical names
        NEW: Tries direct, then legacy aliases, then canonical resolution
        """
        # Direct match
        if skill_id in SEED_PROMPTS:
            return skill_id
        # Try legacy aliases for this canonical ID
        for legacy_key in _CANONICAL_TO_LEGACY.get(skill_id, []):
            if legacy_key in SEED_PROMPTS:
                return legacy_key
        # Try resolving as alias → canonical
        canonical = SKILL_ALIASES.get(skill_id)
        if canonical and canonical in SEED_PROMPTS:
            return canonical
        return None

    def generate(
        self,
        skill_id: str,
        num: int = 50,
        difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed",
    ) -> list[dict]:
        """Generate seed questions for a skill."""

        # OLD: if skill_id not in SEED_PROMPTS — missed canonical IDs like RSN-ARITH
        # NEW: resolve through alias lookup
        prompt_key = self._resolve_prompt_key(skill_id)
        if prompt_key is None:
            print(f"[WARN] No prompt template for {skill_id}, using generic")
            return self._generate_generic(skill_id, num)

        prompt = SEED_PROMPTS[prompt_key].format(num=num)
        
        # Add difficulty modifier
        if difficulty != "mixed" and difficulty in DIFFICULTY_MODIFIERS:
            prompt += DIFFICULTY_MODIFIERS[difficulty]
        
        print(f"[SeedGen] Generating {num} questions for {skill_id}...")
        
        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.8,
        )
        
        # Parse JSON from response
        questions = self._parse_questions(response, skill_id)
        
        print(f"[SeedGen] Generated {len(questions)} questions")
        return questions
    
    def _parse_questions(self, response: str, skill_id: str) -> list[dict]:
        """Parse questions from LLM response."""
        
        # Try to find JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                questions = json.loads(json_match.group())
                # Ensure each has required fields
                result = []
                for i, q in enumerate(questions):
                    if isinstance(q, dict) and "question" in q:
                        q["id"] = f"{skill_id}-SEED-{i+1:04d}"
                        q["skill_bucket"] = skill_id
                        result.append(q)
                    elif isinstance(q, str):
                        result.append({
                            "id": f"{skill_id}-SEED-{i+1:04d}",
                            "question": q,
                            "skill_bucket": skill_id,
                        })
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract numbered questions
        questions = []
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            # Match patterns like "1.", "1)", "- ", "* "
            match = re.match(r'^[\d]+[.\)]\s*(.+)|^[-*]\s*(.+)', line)
            if match:
                q_text = match.group(1) or match.group(2)
                if q_text and len(q_text) > 10:
                    questions.append({
                        "id": f"{skill_id}-SEED-{len(questions)+1:04d}",
                        "question": q_text.strip('"').strip(),
                        "skill_bucket": skill_id,
                    })
        
        return questions
    
    def _generate_generic(self, skill_id: str, num: int) -> list[dict]:
        """Generate questions for unknown skill using generic prompt."""
        
        skill = get_skill_bucket(skill_id)
        
        prompt = f"""Generate {num} diverse questions/problems for testing this skill:

Skill: {skill.name}
Description: {skill.description}
Related benchmarks: {', '.join(skill.related_benchmarks)}

Requirements:
- Questions should be clear and unambiguous
- Varying difficulty levels
- No answers, just questions

Output as JSON array:
[{{"question": "..."}}, ...]"""
        
        response = ollama_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.8,
        )
        
        return self._parse_questions(response, skill_id)
    
    def generate_all(
        self,
        num_per_skill: int = 20,
        skills: list[str] | None = None,
        difficulty: str = "mixed",
    ) -> dict[str, list[dict]]:
        """Generate seeds for multiple skills."""
        
        skills = skills or list(SKILL_BUCKETS.keys())
        all_seeds = {}
        
        for skill_id in skills:
            try:
                seeds = self.generate(skill_id, num_per_skill, difficulty)
                all_seeds[skill_id] = seeds
            except Exception as e:
                print(f"[ERROR] Failed for {skill_id}: {e}")
                all_seeds[skill_id] = []
        
        return all_seeds


# ================================================================
# BUILT-IN SEED TEMPLATES (for offline use)
# ================================================================

BUILTIN_SEEDS = {
    "RSN-ARITHMETIC": [
        {"question": "A store sells apples for $2 each. If you buy 7 apples, how much do you pay?"},
        {"question": "Tom has 45 marbles. He gives 12 to Jane and 8 to Mike. How many marbles does Tom have left?"},
        {"question": "A train travels 240 km in 4 hours. What is its average speed in km/h?"},
        {"question": "If 3/4 of a cake is left and you eat 1/3 of what's left, how much cake remains?"},
        {"question": "A shirt costs $40. It's on sale for 25% off. What is the sale price?"},
        {"question": "Calculate: (15 + 25) × 2 - 30"},
        {"question": "If you have $100 and spend $37.50, how much change do you have?"},
        {"question": "A recipe needs 2.5 cups of flour. How much flour is needed for 3 batches?"},
        {"question": "What is 15% of 80?"},
        {"question": "Divide 156 by 12."},
        {"question": "A car uses 8 liters of fuel per 100 km. How much fuel is needed for 350 km?"},
        {"question": "If 5 workers can complete a job in 12 days, how many days would 10 workers take?"},
        {"question": "The sum of three consecutive numbers is 96. What are the numbers?"},
        {"question": "A rectangle has length 15 cm and width 8 cm. What is its area?"},
        {"question": "Convert 2.75 hours to hours and minutes."},
    ],
    
    "RSN-LOGIC": [
        {"question": "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"},
        {"question": "If it rains, the ground gets wet. The ground is wet. Did it definitely rain?"},
        {"question": "John is older than Mary. Mary is older than Tom. Tom is older than Sue. Who is the oldest?"},
        {"question": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides?"},
        {"question": "If A implies B, and B implies C, and A is true, what can we conclude about C?"},
        {"question": "Some birds can fly. Penguins are birds. Can we conclude that penguins can fly?"},
        {"question": "No reptiles are mammals. All snakes are reptiles. Are any snakes mammals?"},
        {"question": "If today is Monday, then tomorrow is Tuesday. Today is Monday. What is tomorrow?"},
        {"question": "All prime numbers greater than 2 are odd. 7 is a prime number greater than 2. Is 7 odd?"},
        {"question": "Either the door is locked or the window is open. The door is not locked. What can we conclude?"},
        {"question": "If and only if you study, you pass. You didn't pass. Did you study?"},
        {"question": "Box A is heavier than Box B. Box C is lighter than Box B. Which box is heaviest?"},
    ],
    
    "CODE-COMPLETION": [
        {"question": "Complete this function:\ndef factorial(n):\n    '''Return n factorial'''\n    if n <= 1:\n        return 1\n    return"},
        {"question": "Complete this function:\ndef is_palindrome(s):\n    '''Check if string is a palindrome'''\n    s = s.lower()\n    return s =="},
        {"question": "Complete this function:\ndef find_max(numbers):\n    '''Return maximum value in list'''\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for n in numbers:\n        if n > max_val:"},
        {"question": "Complete this function:\ndef count_vowels(s):\n    '''Count vowels in a string'''\n    vowels = 'aeiouAEIOU'\n    count = 0\n    for char in s:\n        if char in vowels:"},
        {"question": "Complete this function:\ndef reverse_list(lst):\n    '''Reverse a list in place'''\n    left = 0\n    right = len(lst) - 1\n    while left < right:"},
        {"question": "Complete this function:\ndef binary_search(arr, target):\n    '''Return index of target or -1'''\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:"},
        {"question": "Complete this function:\ndef fizzbuzz(n):\n    '''Return FizzBuzz result for n'''\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    elif n % 3 == 0:"},
        {"question": "Complete this function:\ndef flatten_list(nested):\n    '''Flatten a nested list'''\n    result = []\n    for item in nested:\n        if isinstance(item, list):"},
    ],
    
    "CODE-DEBUG": [
        {"question": "Fix this code that should print 1 to 10:\nfor i in range(1, 10):\n    print(i)"},
        {"question": "Fix this function that should return True for even numbers:\ndef is_even(n):\n    return n % 2 == 1"},
        {"question": "Fix this code that should reverse a string:\ndef reverse(s):\n    return s[1::-1]"},
        {"question": "Fix this function that should find the average:\ndef average(numbers):\n    return sum(numbers) / len(numbers) + 1"},
        {"question": "Fix this code that should check if a list is sorted:\ndef is_sorted(lst):\n    for i in range(len(lst)):\n        if lst[i] > lst[i+1]:\n            return False\n    return True"},
        {"question": "Fix this recursive function:\ndef countdown(n):\n    print(n)\n    countdown(n-1)"},
    ],
    
    "KNOW-SCIENCE": [
        {"question": "Why does ice float on water instead of sinking?"},
        {"question": "What causes the seasons on Earth?"},
        {"question": "Why is the sky blue during the day?"},
        {"question": "What happens when you mix an acid with a base?"},
        {"question": "Why do we see lightning before we hear thunder?"},
        {"question": "How do vaccines help prevent diseases?"},
        {"question": "Why do metals conduct electricity?"},
        {"question": "What causes tides in the ocean?"},
        {"question": "Why do leaves change color in autumn?"},
        {"question": "How does photosynthesis work?"},
    ],
    
    "KNOW-COMMONSENSE": [
        {"question": "What would happen if you put a metal spoon in a microwave?"},
        {"question": "Why do people wear sunglasses on sunny days?"},
        {"question": "What would happen to a plant if you kept it in complete darkness?"},
        {"question": "Why is it dangerous to text while driving?"},
        {"question": "What would happen if you left milk out of the refrigerator for a week?"},
        {"question": "Why do people typically shake hands when meeting?"},
        {"question": "What would happen if you tried to breathe underwater without equipment?"},
        {"question": "Why do people use bookmarks?"},
        {"question": "What would happen if all the bees disappeared?"},
        {"question": "Why is it impolite to talk with your mouth full?"},
    ],

    # New skill seeds
    "INDIC-QA": [
        {"question": "भारत की राजधानी क्या है?", "language": "hi"},
        {"question": "महात्मा गांधी का जन्म कब हुआ था?", "language": "hi"},
        {"question": "भारत में कितने राज्य हैं?", "language": "hi"},
        {"question": "ताजमहल कहां स्थित है?", "language": "hi"},
        {"question": "भारत का राष्ट्रीय पक्षी कौन सा है?", "language": "hi"},
        {"question": "ভারতের রাজধানী কী?", "language": "bn"},
        {"question": "রবীন্দ্রনাথ ঠাকুর কোথায় জন্মগ্রহণ করেন?", "language": "bn"},
        {"question": "இந்தியாவின் தலைநகரம் என்ன?", "language": "ta"},
        {"question": "భారతదేశ రాజధాని ఏది?", "language": "te"},
    ],

    # OLD: INDIC-TRANS seeds had source_lang/target_lang but no "language" key
    #      → seed.get("language") returned None → fell back to skill.languages[0] = "hi" for all
    # NEW: added "language" key set to target_lang (the output language of the translation)
    "INDIC-TRANS": [
        {"question": "Translate to Hindi: 'Knowledge is power.'", "source_lang": "en", "target_lang": "hi", "language": "hi"},
        {"question": "Translate to Hindi: 'The early bird catches the worm.'", "source_lang": "en", "target_lang": "hi", "language": "hi"},
        {"question": "Translate to English: 'जल ही जीवन है।'", "source_lang": "hi", "target_lang": "en", "language": "en"},
        {"question": "Translate to Bengali: 'Time is money.'", "source_lang": "en", "target_lang": "bn", "language": "bn"},
        {"question": "Translate to Tamil: 'Health is wealth.'", "source_lang": "en", "target_lang": "ta", "language": "ta"},
    ],

    "INDIC-SENT": [
        {"question": "इस वाक्य की भावना क्या है? 'यह फिल्म बहुत बोरिंग थी, समय की बर्बादी।'", "language": "hi"},
        {"question": "इस वाक्य की भावना क्या है? 'बहुत अच्छा अनुभव था, जरूर जाएं!'", "language": "hi"},
        {"question": "Is this positive or negative? 'Product is bakwas, waste of money.'", "language": "mix"},
        {"question": "Is this positive or negative? 'Bahut accha hai yaar, must buy!'", "language": "mix"},
    ],

    "ALN-INST": [
        {"question": "List exactly 5 fruits. Format each as a bullet point. Capitalize the first letter only."},
        {"question": "Write a sentence about dogs. It must have exactly 10 words."},
        {"question": "Count from 1 to 10, but skip all even numbers. Separate with commas."},
        {"question": "Name 3 countries in Europe. Do not mention France or Germany."},
        {"question": "Write 'Hello World' with each word on a new line, in reverse order."},
    ],

    "ALN-STRUCT": [
        {"question": "Return a JSON object with keys 'name', 'age', 'city' for a fictional person."},
        {"question": "Format this data as a markdown table: Apple-$1.50, Banana-$0.75, Orange-$1.00"},
        {"question": "List 3 countries with their capitals in the format 'Country: Capital'"},
        {"question": "Create a JSON array with 3 objects, each having 'id' and 'value' keys."},
    ],

    "PRD-SUM": [
        {"question": "Summarize in one sentence: 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.'"},
        {"question": "What are the main points? 'The meeting covered three topics: budget approval for Q3, new hiring plans for the engineering team, and the product launch timeline for November.'"},
        {"question": "Summarize: 'Climate change is causing global temperatures to rise. This leads to melting ice caps, rising sea levels, and more extreme weather events. Scientists urge immediate action to reduce carbon emissions.'"},
    ],

    # OLD: LANG-TRANS seeds had no "language" key → all got "en" (skill.languages[0])
    # NEW: added "language" key set to target language of the translation
    "LANG-TRANS": [
        {"question": "Translate to Hindi: 'The book is on the table.'", "language": "hi"},
        {"question": "Translate to English: 'मुझे हिंदी में बात करना पसंद है।'", "language": "en"},
        {"question": "Translate to Hindi: 'Time waits for no one.'", "language": "hi"},
        {"question": "Translate to English: 'विद्या सबसे बड़ा धन है।'", "language": "en"},
    ],

    "CODE-GEN-T1": [
        {"question": "Write a Python function to check if a number is even."},
        {"question": "Write a Python function to find the maximum of two numbers."},
        {"question": "Write a Python function to reverse a string."},
        {"question": "Write a Python function to count vowels in a string."},
        {"question": "Write a Python function to calculate the factorial of a number."},
    ],
}


# OLD: BUILTIN_SEEDS keys are legacy names (RSN-ARITHMETIC), but generate-bank
#      iterates canonical names (RSN-ARITH) — most skills got garbage placeholders.
# NEW: Build a reverse alias map so canonical IDs can find legacy BUILTIN_SEEDS keys.
from common.skills import SKILL_ALIASES

# Reverse map: canonical → list of legacy aliases  (e.g. "RSN-ARITH" → ["RSN-ARITHMETIC"])
_CANONICAL_TO_LEGACY: dict[str, list[str]] = {}
for _legacy, _canonical in SKILL_ALIASES.items():
    _CANONICAL_TO_LEGACY.setdefault(_canonical, []).append(_legacy)


def get_builtin_seeds(skill_id: str, num: int = 10) -> list[dict]:
    """Get built-in seed questions (no LLM needed).

    OLD: Only looked up skill_id directly in BUILTIN_SEEDS — missed canonical IDs
    NEW: Tries skill_id first, then resolves canonical→legacy aliases to find seeds
    """

    seeds = None

    # 1. Direct lookup (works for legacy keys like "RSN-ARITHMETIC")
    if skill_id in BUILTIN_SEEDS:
        seeds = BUILTIN_SEEDS[skill_id][:num]

    # 2. NEW: Try legacy aliases for this canonical ID
    #    e.g. skill_id="RSN-ARITH" → look up "RSN-ARITHMETIC" in BUILTIN_SEEDS
    if seeds is None:
        for legacy_key in _CANONICAL_TO_LEGACY.get(skill_id, []):
            if legacy_key in BUILTIN_SEEDS:
                seeds = BUILTIN_SEEDS[legacy_key][:num]
                break

    # 3. NEW: Try resolving skill_id as an alias → canonical → check BUILTIN_SEEDS
    #    e.g. skill_id="RSN-ARITHMETIC" alias→ "RSN-ARITH" → check BUILTIN_SEEDS["RSN-ARITH"]
    if seeds is None:
        canonical = SKILL_ALIASES.get(skill_id)
        if canonical and canonical in BUILTIN_SEEDS:
            seeds = BUILTIN_SEEDS[canonical][:num]

    # 4. Fallback: generate placeholders (unchanged)
    if seeds is None:
        seeds = [
            {"question": f"Sample question {i+1} for {skill_id}"}
            for i in range(num)
        ]

    # Add metadata
    for i, s in enumerate(seeds):
        s["id"] = f"{skill_id}-SEED-{i+1:04d}"
        s["skill_bucket"] = skill_id

    return seeds


# ================================================================
# CLI
# ================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate seed questions for synthetic data"
    )
    parser.add_argument("--skill", "-s", help="Skill bucket ID")
    parser.add_argument("--all", action="store_true", help="Generate for all skills")
    parser.add_argument("--num", "-n", type=int, default=20, help="Questions per skill")
    parser.add_argument("--model", "-m", default="qwen3:8b", help="Ollama model")
    parser.add_argument("--difficulty", "-d", default="mixed",
                        choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--output", "-o", help="Output file (JSONL)")
    parser.add_argument("--builtin", action="store_true", 
                        help="Use built-in seeds (no LLM)")
    parser.add_argument("--list", action="store_true", help="List available skills")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable skill buckets:")
        for skill_id, skill in SKILL_BUCKETS.items():
            has_prompt = "✓" if skill_id in SEED_PROMPTS else "○"
            has_builtin = "✓" if skill_id in BUILTIN_SEEDS else "○"
            print(f"  {has_prompt}{has_builtin} {skill_id:20s} {skill.name}")
        print("\n  ✓ = has template, ○ = generic/none")
        return
    
    if not args.skill and not args.all:
        parser.print_help()
        return
    
    all_seeds = []
    
    if args.builtin:
        # Use built-in seeds
        skills = list(SKILL_BUCKETS.keys()) if args.all else [args.skill]
        for skill_id in skills:
            seeds = get_builtin_seeds(skill_id, args.num)
            all_seeds.extend(seeds)
            print(f"[Builtin] {skill_id}: {len(seeds)} seeds")
    else:
        # Generate with LLM
        generator = SeedGenerator(model=args.model)
        
        if args.all:
            result = generator.generate_all(args.num, difficulty=args.difficulty)
            for skill_id, seeds in result.items():
                all_seeds.extend(seeds)
        else:
            seeds = generator.generate(args.skill, args.num, args.difficulty)
            all_seeds.extend(seeds)
    
    # Output
    output_path = args.output or f"seeds_{datetime.now():%Y%m%d_%H%M}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for seed in all_seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    
    print(f"\n[Done] Saved {len(all_seeds)} seeds to: {output_path}")
    
    # Show sample
    if all_seeds:
        print("\nSample questions:")
        for seed in all_seeds[:3]:
            q = seed["question"][:80] + "..." if len(seed["question"]) > 80 else seed["question"]
            print(f"  - {q}")


if __name__ == "__main__":
    main()
