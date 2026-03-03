# Group 2: Math and Numbers - Curriculum Learning Distribution

## Overview
This document explains the curriculum learning distribution for Group 2 (Math and Numbers), which generates **600,000 samples** across 6 statement types.

## Total Distribution: 600,000 Samples

| Statement | Type | Samples | Percentage | Rationale |
|-----------|------|---------|------------|-----------|
| **S1** | Counting Sequences | 60,000 | 10% | Foundation for number understanding |
| **S2** | Before/After | 80,000 | 13.3% | Number sequence comprehension |
| **S3** | Word Problems | 120,000 | 20% | Real-world application of arithmetic |
| **S4** | Number Comparison | 100,000 | 16.7% | Critical for arithmetic reasoning |
| **S5** | Direct Math Queries | 150,000 | 25% | Core arithmetic operations |
| **S6** | Word-Based Math | 90,000 | 15% | Linguistic bridge to math |

## Curriculum Learning Progression

### Phase 1: Number Fundamentals (140,000 samples - 23.3%)
**S1 + S2: Counting and Sequencing**

#### Why First?
- **S1 (Counting)**: Most basic skill - understanding number order
- **S2 (Before/After)**: Builds directly on counting - understanding adjacency

#### Learning Objectives:
- Establish number sequence 1-100 (and beyond)
- Understand successor/predecessor relationships
- Handle positive, negative, and zero

---

### Phase 2: Number Relationships (100,000 samples - 16.7%)
**S4: Number Comparison**

#### Why Second?
- Requires understanding from Phase 1 (what numbers mean)
- Prerequisite for arithmetic (need to understand magnitude)
- Foundation for inequality reasoning

#### Learning Objectives:
- Compare positive vs positive numbers
- Compare negative vs negative numbers
- Compare across positive/negative/zero
- Understand "greater than", "less than", "equal"

---

### Phase 3: Core Arithmetic (150,000 samples - 25%)
**S5: Direct Mathematical Queries**

#### Why Third?
- Builds on number understanding and comparison
- Most fundamental arithmetic skill
- Highest sample allocation due to:
  - 4 operations (+, -, ×, ÷)
  - 2-term, 3-term, 4-term complexity
  - Largest variation potential

#### Learning Objectives:
- Master basic operations (+, -, ×, ÷)
- Understand order of operations (BODMAS/PEMDAS)
- Handle multi-term expressions
- Work with negative results and decimals

---

### Phase 4: Applied Arithmetic (120,000 samples - 20%)
**S3: Word Problems with Objects**

#### Why Fourth?
- Applies arithmetic skills from Phase 3
- Adds contextual complexity (objects, scenarios)
- Bridges abstract math to real-world situations

#### Learning Objectives:
- Extract arithmetic from natural language
- Apply operations in context (apples, toys, etc.)
- Handle multi-step problems
- Map scenarios to operations

---

### Phase 5: Advanced Integration (90,000 samples - 15%)
**S6: Word-Based Mathematical Queries**

#### Why Last?
- Most complex: combines language understanding + arithmetic
- Requires all previous skills
- Abstract linguistic constructs ("more than", "less than", "double")

#### Learning Objectives:
- Parse complex linguistic math expressions
- Map phrases to operations ("4 less than 10" → 10 - 4)
- Handle nested operations ("double of 5 plus 3")
- Bridge natural language and mathematical notation

---

## Detailed Breakdown by Statement

### S1: Counting Sequences (60,000 samples)

#### Distribution by Type:
| Type | Samples | Percentage | Examples |
|------|---------|------------|----------|
| Count from 1 | 42,000 | 70% | "Count till 10" → "1, 2, 3, ..., 10" |
| Custom start/end | 18,000 | 30% | "Count from 5 to 8" → "5, 6, 7, 8" |

#### Distribution by Difficulty:
| Difficulty | Count Range | Samples | Percentage | Rationale |
|------------|-------------|---------|------------|-----------|
| Easy | 1-10 or small ranges (≤5) | 24,000 | 40% | Most frequent in early learning |
| Medium | 11-30 or medium ranges (6-15) | 24,000 | 40% | Transition to higher numbers |
| Hard | 31-100 or large ranges (16+) | 12,000 | 20% | Challenge & completeness |

#### Sample Complexity Progression:
- **Easy**: "Can you count till 5?" → "1, 2, 3, 4, 5"
- **Easy (custom)**: "Count from 3 to 5" → "3, 4, 5"
- **Medium**: "Count from 1 to 20" → "1, 2, 3, ..., 19, 20"
- **Medium (custom)**: "Numbers from 10 to 20" → "10, 11, 12, ..., 20"
- **Hard**: "List numbers up to 75" → "1, 2, 3, ..., 74, 75"
- **Hard (custom)**: "Count from 50 to 70" → "50, 51, 52, ..., 70"

---

### S2: Before/After (80,000 samples)

#### Distribution by Query Type:
| Type | Samples | Percentage |
|------|---------|------------|
| After queries | 40,000 | 50% |
| Before queries | 40,000 | 50% |

#### Distribution by Format:
| Format | Samples | Percentage | Examples |
|--------|---------|------------|----------|
| Single number | 48,000 | 60% | "What comes after 5?" → "6" |
| Multiple numbers (window 2-5) | 32,000 | 40% | "What are the next 3 numbers after 10?" → "11, 12, 13" |

#### Distribution by Number Range (within each type):
| Range | Percentage | Rationale |
|-------|------------|-----------|
| Small positive (1-100) | 40% | Most common range |
| Large positive (101-1000) | 30% | Extended range |
| Negative (-100 to -1) | 20% | Introduce negatives |
| Edge cases (0, boundaries) | 10% | Special handling |

#### Sample Complexity Progression:
- **Easy (single)**: "What comes after 5?" → "6"
- **Easy (multi)**: "What are the next 2 numbers after 5?" → "6, 7"
- **Medium (single)**: "What comes before 250?" → "249"
- **Medium (multi)**: "List 3 numbers that come before 100" → "99, 98, 97"
- **Hard (single)**: "What is the number after -7?" → "-6"
- **Hard (multi)**: "Give me 4 numbers following -10" → "-9, -8, -7, -6"

---

### S3: Word Problems with Objects (120,000 samples)

#### Distribution by Operation:
| Operation | Samples | Percentage | Rationale |
|-----------|---------|------------|-----------|
| Addition | 50,000 | 41.7% | Most fundamental operation |
| Subtraction | 40,000 | 33.3% | Second fundamental operation |
| Mixed (2+ ops) | 30,000 | 25% | Multi-step reasoning |

#### Object Categories:
- Fruits: apples, oranges, bananas, mangoes, grapes
- Animals: cats, dogs, birds, fish, rabbits
- Toys: balls, dolls, cars, blocks, puzzles
- Food: cookies, candies, chocolates, pizzas
- School: pencils, books, erasers, notebooks
- Nature: flowers, leaves, stones, shells

#### Sample Complexity Progression:
- **Easy**: "If you have 2 apples and get 3 more, how many apples now?" → "5"
- **Medium**: "You have 15 cookies. If you eat 7, how many are left?" → "8"
- **Hard**: "Start with 10 marbles, give away 3, find 5 more. How many now?" → "12"

---

### S4: Number Comparison (100,000 samples)

#### Distribution by Query Type:
| Type | Samples | Percentage | Rationale |
|------|---------|------------|-----------|
| Greater queries | 45,000 | 45% | Core comparison |
| Smaller queries | 45,000 | 45% | Inverse comparison |
| Equal cases | 10,000 | 10% | Edge case handling |

#### Distribution by Number Type (within comparison types):
| Number Type | Percentage | Examples |
|-------------|------------|----------|
| Positive vs Positive | 30% | "5 vs 10", "100 vs 99" |
| Negative vs Negative | 25% | "-5 vs -10", "-2 vs -7" |
| Positive vs Negative | 25% | "5 vs -9", "3 vs -7" |
| Zero comparisons | 10% | "0 vs 5", "0 vs -3" |
| Large numbers | 10% | "1000 vs 999", "5000 vs 5001" |

#### Sample Complexity Progression:
- **Easy**: "Which is bigger, 5 or 3?" → "5"
- **Medium**: "Between -5 and -10, which is greater?" → "-5"
- **Hard**: "Which number is smaller, 7 or 7?" → "equal"

---

### S5: Direct Mathematical Queries (150,000 samples)

#### Distribution by Operation Type:
| Operation | Samples | Percentage | Rationale |
|-----------|---------|------------|-----------|
| 2-term Addition | 30,000 | 20% | Basic addition |
| 2-term Subtraction | 25,000 | 16.7% | Basic subtraction |
| 2-term Multiplication | 25,000 | 16.7% | Times tables |
| 2-term Division | 20,000 | 13.3% | Basic division |
| 3-term Mixed | 30,000 | 20% | Multi-step with BODMAS |
| 4-term Mixed | 20,000 | 13.3% | Complex BODMAS |

#### Complexity Features:
- **Order of Operations**: Follow BODMAS/PEMDAS for 3-4 term expressions
- **Decimals**: Allow simple decimals for division (e.g., 5/2 = 2.5)
- **Negative Results**: Handle negative outcomes (e.g., 3 - 8 = -5)
- **Special Cases**: Zero operations, identity operations

#### Sample Complexity Progression:
- **Easy**: "What is 5 + 3?" → "8"
- **Medium**: "What is 10 - 4 + 2?" → "8"
- **Hard**: "What is 2 + 3 × 4 - 1?" → "13" (BODMAS: 2 + 12 - 1)

---

### S6: Word-Based Mathematical Queries (90,000 samples)

#### Distribution by Operation Type:
| Type | Samples | Percentage | Rationale |
|------|---------|------------|-----------|
| "More than" (addition) | 25,000 | 27.8% | Most common phrase |
| "Less than" (subtraction) | 25,000 | 27.8% | Inverse of "more than" |
| "Times/Double/Triple" | 15,000 | 16.7% | Multiplication phrases |
| "Half of/Quarter of" | 10,000 | 11.1% | Division phrases |
| Complex phrases | 15,000 | 16.7% | Multi-operation linguistic |

#### Linguistic Complexity:
- **Simple**: "What is 3 more than 7?" → "10"
- **Moderate**: "What is double of 8?" → "16"
- **Complex**: "What is 5 more than the double of 3?" → "11" (3×2 + 5)

#### Key Phrases Mapping:
- "X more than Y" → Y + X
- "X less than Y" → Y - X
- "double/twice X" → X × 2
- "triple/thrice X" → X × 3
- "half of X" → X ÷ 2
- "quarter of X" → X ÷ 4

---

## Curriculum Learning Benefits

### 1. Progressive Complexity
- Start with counting (most basic)
- Build to comparison (understanding magnitude)
- Move to arithmetic (operations)
- Apply to context (word problems)
- Master language integration (word-based math)

### 2. Skill Scaffolding
Each phase builds on previous:
- **S1 → S2**: Counting enables before/after
- **S1, S2 → S4**: Sequence understanding enables comparison
- **S4 → S5**: Comparison understanding enables arithmetic
- **S5 → S3**: Arithmetic enables word problems
- **S3, S5 → S6**: Both enable linguistic math

### 3. Balanced Coverage
- Foundation (S1, S2): 23.3% - adequate for basic skills
- Core (S5): 25% - largest allocation for most important skill
- Application (S3, S4, S6): 51.7% - majority in applied contexts

### 4. Variation Potential
Distribution reflects capacity:
- **S5 (150k)**: Highest variation (4 ops × terms × ranges)
- **S3 (120k)**: High variation (objects × operations)
- **S4 (100k)**: High variation (number pairs × comparisons)
- **S6 (90k)**: Moderate variation (phrases × operations)
- **S2 (80k)**: Moderate variation (numbers × before/after)
- **S1 (60k)**: Limited variation (100 count targets)

---

## Implementation Notes

### Difficulty Distribution Philosophy
- **Easy (30-40%)**: Build confidence, cover basics
- **Medium (40-50%)**: Main learning zone
- **Hard (20%)**: Challenge and extend

### Template Diversity
Each statement type has 10-20 semantic variations to prevent overfitting to specific phrasings.

### Deduplication Strategy
Using dictionary keys ensures no duplicate queries, even across different generators.

### Validation Strategy
Pattern-based categorization validates actual distribution matches expected distribution.

---

## Expected Learning Outcomes

After training on Group 2, a model should:
1. Count from 1 to 100 reliably
2. Understand number sequences (before/after)
3. Compare numbers including negatives
4. Perform basic arithmetic (+, -, ×, ÷)
5. Handle multi-term expressions with BODMAS
6. Solve word problems with objects
7. Parse linguistic math expressions
8. Bridge natural language and mathematical operations

---

## Comparison to Group 1

| Aspect | Group 1 (Language) | Group 2 (Math) |
|--------|-------------------|----------------|
| Total Samples | 700,000 | 600,000 |
| Statement Types | 10 | 6 |
| Domain | Literacy | Numeracy |
| Complexity | Low-Medium | Low-Medium |
| Priority | Critical (35%) | Critical (30%) |
| Variation | Very High (words) | High (numbers) |

Both groups are foundational, with Group 1 slightly larger due to higher word variation potential.
