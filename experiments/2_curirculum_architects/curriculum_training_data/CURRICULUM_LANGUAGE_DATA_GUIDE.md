# Curriculum Language Data Generation Guide

**Comprehensive Formatting Rules for Language and Literacy Training Data**

This guide provides detailed formatting rules and best practices for generating curriculum language training datasets. Apply these rules during dataset generation or as post-processing steps to ensure consistency, correctness, and quality across all language-specific training data.

---

## Table of Contents

1. [Dataset Format Structure](#dataset-format-structure)
2. [Question Types and Patterns](#question-types-and-patterns)
3. [Word Quoting Rules](#word-quoting-rules)
4. [Answer Format Guidelines](#answer-format-guidelines)
5. [Language-Specific Considerations](#language-specific-considerations)
6. [Implementation Patterns](#implementation-patterns)
7. [Edge Cases and Special Scenarios](#edge-cases-and-special-scenarios)
8. [Quality Assurance](#quality-assurance)
9. [Common Mistakes and Fixes](#common-mistakes-and-fixes)

---

## Dataset Format Structure

### Format Pattern

The dataset follows a simple, continuous question-answer format:

**Format:** `Q? A. Q? A. Q? A. ...`

Where:
- **Q** = Question (must end with `?`)
- **A** = Answer (must end with punctuation mark)
- Space after question mark (`? `)
- Answer ends with language-specific punctuation:
  - **English:** Period (`.`) - `Q? A. Q? A. ...`
  - **Hindi:** Devanagari danda (`।`) - `Q? A। Q? A। ...`
- Each `Q?A` pair is separated by punctuation + space (`. ` for English, `। ` for Hindi)
- Multiple Q?A pairs appear on the same line
- No line breaks between Q?A pairs within the dataset

### Detailed Format Requirements

1. **Question Format:**
   - Must end with a question mark (`?`)
   - Can contain quoted words (target words wrapped in double quotes)
   - Can contain punctuation marks (commas, colons, etc.) as needed
   - Should be grammatically correct and clear

2. **Answer Format:**
   - Must end with language-specific punctuation:
     - **English:** Period (`.`)
     - **Hindi:** Devanagari danda (`।`)
   - For spelling questions: comma-separated letters (e.g., `c, a, t`)
   - For letter position questions: single letter (e.g., `c`)
   - For letter count questions: numeric value (e.g., `3`)
   - For sound/choice questions: the selected word (e.g., `chair`)

3. **Pair Separation:**
   - **English:** Exactly one period (`.`) followed by exactly one space (` `)
   - **Hindi:** Exactly one Devanagari danda (`।`) followed by exactly one space (` `)
   - No additional punctuation between pairs
   - No line breaks or newlines

### Examples from `output/group1.txt`

**✅ Correct Format (English):**

Format: `Q? A. Q? A. Q? A. ...`

```
What is the spelling of "behavior"? b, e, h, a, v, i, o, r. What's the spelling of "curry"? c, u, r, r, y. Write the spelling of "sneak"? s, n, e, a, k. Can you spell "teens"? t, e, e, n, s. Tell me the spelling of "wha"? w, h, a.
```

```
What are the letters in "file"? f, i, l, e. What is the spelling of "hole"? h, o, l, e. Tell me the spelling of "boost"? b, o, o, s, t. What is the spelling of "classify"? c, l, a, s, s, i, f, y.
```

```
Break down "bit" into letters? b, i, t. Show me the spelling of "okay"? o, k, a, y. What is the spelling of "sake"? s, a, k, e. Tell me the spelling of "venal"? v, e, n, a, l.
```

**✅ Correct Format (Hindi):**

Format: `Q? A। Q? A। Q? A। ...` (Note: Hindi uses Devanagari danda `।` instead of period `.`)

```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र। "पानी" की वर्तनी क्या है? प, आ, न, ी। "सूरज" की वर्तनी क्या है? स, ू, र, ज। "विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
```

```
"फूल" की वर्तनी क्या है? फ, ू, ल। "किताब" की वर्तनी क्या है? क, ि, त, ा, ब। "बच्चा" की वर्तनी क्या है? ब, च, ्, च, ा। "स्कूल" की वर्तनी क्या है? स, ्, क, ू, ल।
```

```
"कमल" का पहला अक्षर क्या है? क। "कमल" का दूसरा अक्षर क्या है? म। "पानी" में कितने अक्षर हैं? 4। "घर" में कितने अक्षर हैं? 2।
```

**❌ Incorrect Format:**

- **Missing punctuation after answer:** `What is the spelling of "behavior"? b, e, h, a, v, i, o, r What's the spelling of "curry"?`
- **Missing question mark:** `What is the spelling of "behavior" b, e, h, a, v, i, o, r.`
- **Missing space after question mark:** `What is the spelling of "behavior"?b, e, h, a, v, i, o, r.` (should be `? `)
- **Missing space between pairs:** `What is the spelling of "behavior"? b, e, h, a, v, i, o, r.What's the spelling of "curry"?`
- **Wrong punctuation for Hindi:** `"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र।` (should use `।` not `.`)
- **Extra line breaks:** Each Q?A pair should be on the same line, separated only by `. ` (English) or `। ` (Hindi)
- **Multiple spaces:** `What is the spelling of "behavior"? b, e, h, a, v, i, o, r.  What's the spelling of "curry"?` (should be single space)

---

## Question Types and Patterns

### 1. Spelling Questions

These questions ask for the complete spelling of a word.

**Common Patterns:**
- `What is the spelling of "word"?`
- `What's the spelling of "word"?`
- `Can you spell "word"?`
- `Tell me the spelling of "word"?`
- `Write the spelling of "word"?`
- `Show me the spelling of "word"?`
- `Provide the spelling of "word"?`
- `Give me the spelling of "word"?`
- `Spell "word"?`
- `Spell out "word"?`
- `How do you spell "word"?`
- `What does "word" spell?`
- `What is "word" spelled as?`
- `Break down "word" into letters?`

**Examples (English):**
```
What is the spelling of "behavior"? b, e, h, a, v, i, o, r. What's the spelling of "curry"? c, u, r, r, y. Can you spell "teens"? t, e, e, n, s.
```

**Examples (Hindi):**
```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र। "पानी" की वर्तनी क्या है? प, आ, न, ी।
```

**Answer Format:** Comma-separated letters: `b, e, h, a, v, i, o, r` (English) or `क, म, ल` (Hindi)

### 2. Letter Position Questions

These questions ask for a specific letter at a given position in a word.

**Common Patterns:**
- `What is the first letter in "word"?`
- `What is the second letter in "word"?`
- `What is the third letter in "word"?`
- `What is the [N]th letter in "word"?`
- `What is the [N] letter in "word"?`
- `Tell me the [N] letter of "word"?`
- `Give me the [N] letter of "word"?`
- `Find the [N] letter of "word"?`
- `Identify the [N] letter of "word"?`
- `Can you say the [N] letter in "word"?`
- `Which letter is at position [N] in "word"?`
- `What's the [N]th letter in "word"?`
- `What's the [N] letter in "word"?`

**Examples (English):**
```
What is the first letter in "apple"? a. What is the second letter in "apple"? p. Tell me the third letter of "apple"? p. Give me the 4th letter of "apple"? l.
```

**Examples (Hindi):**
```
"कमल" का पहला अक्षर क्या है? क। "कमल" का दूसरा अक्षर क्या है? म। "कमल" का तीसरा अक्षर क्या है? ल। "पानी" का पहला अक्षर क्या है? प।
```

**Answer Format:** Single letter: `a`, `p`, `p`, `l` (English) or `क`, `म`, `ल` (Hindi)

**Note:** Position can be specified as ordinal (first, second, third) or numeric (1, 2, 3, 4th, etc.)

### 3. Letter Count Questions

These questions ask for the total number of letters in a word.

**Common Patterns:**
- `How many letters are in "word"?`
- `How many letters does "word" have?`
- `Count the number of letters in "word"?`
- `Count the letters in "word"?`
- `Find the number of letters in "word"?`
- `Tell me the number of letters in "word"?`
- `What is the total letter count for "word"?`
- `What's the letter count of "word"?`
- `Give me the letter count of "word"?`
- `How long is the word "word"?`
- `What is the length of word "word"?`
- `How many alphabets are there in "word"?`
- `Letter count of "word"?`

**Examples (English):**
```
How many letters are in "cat"? 3. Count the number of letters in "behavior"? 8. What is the total letter count for "curry"? 5.
```

**Examples (Hindi):**
```
"कमल" में कितने अक्षर हैं? 3। "घर" में कितने अक्षर हैं? 2। "पानी" में कितने अक्षर हैं? 4। "विद्यालय" में कितने अक्षर हैं? 8।
```

**Answer Format:** Numeric value: `3`, `8`, `5` (same for all languages)

### 4. Letter Listing Questions

These questions ask for all letters in a word (similar to spelling but phrased differently).

**Common Patterns:**
- `What are the letters in "word"?`
- `Break down "word" into letters?`

**Examples (English):**
```
What are the letters in "file"? f, i, l, e. Break down "bit" into letters? b, i, t.
```

**Examples (Hindi):**
```
"कमल" के अक्षर क्या हैं? क, म, ल। "घर" को अक्षरों में तोड़ें? घ, र। "पानी" के अक्षर क्या हैं? प, आ, न, ी।
```

**Answer Format:** Comma-separated letters: `f, i, l, e` (English) or `क, म, ल` (Hindi)

### 5. Sound Matching Questions

These questions ask to identify words that start with specific sounds.

**Common Patterns:**
- `Tell me which word starts with /sound/: "word1" or "word2"?`
- `Choose the word with starting sound /sound/: "word1" or "word2"?`
- `Name the word that starts with /sound/: "word1" or "word2"?`
- `Pick the word that begins with sound /sound/: "word1" or "word2"?`
- `Which of these begins with /sound/: "word1" or "word2"?`
- `Find the word starting with /sound/: "word1" or "word2"?`
- `What word begins with /sound/, "word1" or "word2"?`
- `Which word has the initial sound /sound/: "word1" or "word2"?`
- `Say which word has the /sound/ sound at the start: "word1" or "word2"?`

**Examples:**
```
Tell me which word starts with /ch/: "dog" or "chair"? chair. Choose the word with starting sound /p/: "blame" or "party"? party. Which word has the initial sound /sm/: "dog" or "smell"? smell.
```

**Answer Format:** The selected word: `chair`, `party`, `smell`

**Note:** Sound notation uses forward slashes (e.g., `/ch/`, `/p/`, `/th/`)

### 6. Language-Specific Question Patterns

Different languages may have different question patterns. Here are examples:

**Hindi (Devanagari Script):**

**Spelling Questions:**
- `"कमल" की वर्तनी क्या है?` (What is the spelling of "kamal"?)
- `"घर" की वर्तनी क्या है?` (What is the spelling of "ghar"?)
- `"पानी" की वर्तनी क्या है?` (What is the spelling of "pani"?)
- `"सूरज" की वर्तनी क्या है?` (What is the spelling of "suraj"?)
- `"विद्यालय" की वर्तनी क्या है?` (What is the spelling of "vidyalay"?)
- `"फूल" की वर्तनी क्या है?` (What is the spelling of "phool"?)
- `"किताब" की वर्तनी क्या है?` (What is the spelling of "kitab"?)
- `"बच्चा" की वर्तनी क्या है?` (What is the spelling of "baccha"?)
- `"स्कूल" की वर्तनी क्या है?` (What is the spelling of "school"?)
- `"गाड़ी" की वर्तनी क्या है?` (What is the spelling of "gaadi"?)

**Letter Position Questions:**
- `"कमल" का पहला अक्षर क्या है?` (What is the first letter of "kamal"?)
- `"कमल" का दूसरा अक्षर क्या है?` (What is the second letter of "kamal"?)
- `"कमल" का तीसरा अक्षर क्या है?` (What is the third letter of "kamal"?)
- `"पानी" का पहला अक्षर क्या है?` (What is the first letter of "pani"?)
- `"विद्यालय" का चौथा अक्षर क्या है?` (What is the fourth letter of "vidyalay"?)

**Letter Count Questions:**
- `"कमल" में कितने अक्षर हैं?` (How many letters are in "kamal"?)
- `"घर" में कितने अक्षर हैं?` (How many letters are in "ghar"?)
- `"पानी" में कितने अक्षर हैं?` (How many letters are in "pani"?)
- `"विद्यालय" में कितने अक्षर हैं?` (How many letters are in "vidyalay"?)
- `"बच्चा" में कितने अक्षर हैं?` (How many letters are in "baccha"?)

**Letter Listing Questions:**
- `"कमल" के अक्षर क्या हैं?` (What are the letters in "kamal"?)
- `"घर" को अक्षरों में तोड़ें?` (Break down "ghar" into letters?)
- `"पानी" के अक्षर क्या हैं?` (What are the letters in "pani"?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `क, म, ल` or `घ, र` or `प, आ, न, ी`
- Letter Position: Single character: `क`, `म`, `ल`
- Letter Count: Numeric value: `3`, `2`, `4`

---

## Word Quoting Rules

### Core Rule

**Wrap target words in double quotes (`"`) when they appear in spelling or language-related questions.**

### When to Quote

Quote the word being asked about in:
- Spelling questions
- Letter position questions
- Letter count questions
- Letter listing questions
- Sound matching questions (quote all candidate words)
- Any language-related question where a specific word is the focus

### When NOT to Quote

Do NOT quote:
- Answer letters/characters
- Answer numbers
- Answer words (in sound matching questions, the answer is the selected word, not quoted)
- Words that are already quoted
- Words in explanatory text or context

### Examples

**✅ Correct Quoting:**

```
What is the spelling of "behavior"? b, e, h, a, v, i, o, r.
What is the first letter in "apple"? a.
How many letters are in "cat"? 3.
Tell me which word starts with /ch/: "dog" or "chair"? chair.
"कमल" की वर्तनी क्या है? क, म, ल।
"घर" की वर्तनी क्या है? घ, र।
"पानी" की वर्तनी क्या है? प, आ, न, ी।
"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
"कमल" का पहला अक्षर क्या है? क।
"पानी" में कितने अक्षर हैं? 4।
```

**❌ Incorrect Quoting:**

```
What is the spelling of behavior? → Should quote: "behavior"
What is the spelling of "cat"? "c", "a", "t" → Answer letters should NOT be quoted
Tell me which word starts with /ch/: dog or "chair"? → Both words should be quoted
What is the spelling of ""cat""? → Double-quoting (word already quoted)
```

### Quoting Patterns by Question Type

1. **Spelling Questions:**
   - Quote the target word: `What is the spelling of "word"?`
   - Answer: Unquoted comma-separated letters: `w, o, r, d`

2. **Letter Position Questions:**
   - Quote the target word: `What is the first letter in "word"?`
   - Answer: Single unquoted letter: `w`

3. **Letter Count Questions:**
   - Quote the target word: `How many letters are in "word"?`
   - Answer: Unquoted number: `4`

4. **Sound Matching Questions:**
   - Quote all candidate words: `Tell me which word starts with /ch/: "dog" or "chair"?`
   - Answer: Unquoted selected word: `chair`

5. **Language-Specific Questions:**
   - Quote the target word in the appropriate script: `"कमल" की वर्तनी क्या है?`
   - Answer: Unquoted comma-separated characters: `क, म, ल`
   - More Hindi examples:
     - `"घर" की वर्तनी क्या है? घ, र।`
     - `"पानी" की वर्तनी क्या है? प, आ, न, ी।`
     - `"सूरज" की वर्तनी क्या है? स, ू, र, ज।`
     - `"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।`
     - `"कमल" का पहला अक्षर क्या है? क।`
     - `"पानी" में कितने अक्षर हैं? 4।`

---

## Answer Format Guidelines

### Important: Language-Specific Punctuation

**All answers must end with language-specific punctuation:**
- **English:** Period (`.`) - Example: `c, a, t.`
- **Hindi:** Devanagari danda (`।`) - Example: `क, म, ल।`

**Note:** The punctuation mark appears after the answer content and before the space that separates Q?A pairs.

### Spelling Answers

**Format:** Comma-separated letters with spaces after commas

**Examples:**
- `c, a, t` (3 letters)
- `b, e, h, a, v, i, o, r` (8 letters)
- `c, u, r, r, y` (5 letters)

**Rules:**
- Each letter separated by comma and space: `, `
- No quotes around individual letters
- No quotes around the entire answer
- Preserve case (usually lowercase for common words)
- Handle special characters correctly (e.g., hyphens, apostrophes)

### Letter Position Answers

**Format:** Single letter

**Examples:**
- `a` (first letter of "apple")
- `p` (second letter of "apple")
- `l` (fourth letter of "apple")

**Rules:**
- Single letter only
- No quotes
- Preserve case (usually lowercase)

### Letter Count Answers

**Format:** Numeric value

**Examples:**
- `3` (for "cat")
- `8` (for "behavior")
- `5` (for "curry")

**Rules:**
- Numeric digits only
- No quotes
- No units or labels (just the number)

### Sound Matching Answers

**Format:** The selected word (unquoted)

**Examples:**
- `chair` (selected from "dog" or "chair")
- `party` (selected from "blame" or "party")
- `smell` (selected from "dog" or "smell")

**Rules:**
- The complete word
- No quotes
- Preserve case (usually lowercase)

### Language-Specific Answer Formats

**Hindi (Devanagari):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `क, म, ल।`
- Preserve script-specific formatting
- Handle combining characters correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"कमल" की वर्तनी क्या है? क, म, ल।` (Note: ends with `।`)
  - `"घर" की वर्तनी क्या है? घ, र।`
  - `"पानी" की वर्तनी क्या है? प, आ, न, ी।`
  - `"सूरज" की वर्तनी क्या है? स, ू, र, ज।`
  - `"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।`
  - `"फूल" की वर्तनी क्या है? फ, ू, ल।`
  - `"किताब" की वर्तनी क्या है? क, ि, त, ा, ब।`
  - `"बच्चा" की वर्तनी क्या है? ब, च, ्, च, ा।`
  - `"स्कूल" की वर्तनी क्या है? स, ्, क, ू, ल।`
  - `"गाड़ी" की वर्तनी क्या है? ग, ा, ड, ्, र, ी।`
  - `"कमल" का पहला अक्षर क्या है? क।` (Letter position - ends with `।`)
  - `"पानी" में कितने अक्षर हैं? 4।` (Letter count - ends with `।`)

---

## Language-Specific Considerations

### Unicode Handling

**Critical Requirements:**
- Preserve all Unicode characters correctly
- Don't break multi-byte characters
- Handle combining characters (diacritics) properly
- Maintain script-specific formatting

**Scripts to Handle:**
- **Devanagari (Hindi):** `"कमल"`, `"घर"`, `"पानी"`, `"सूरज"`, `"विद्यालय"`, `"फूल"`, `"किताब"`, `"बच्चा"`, `"स्कूल"`, `"गाड़ी"`
- **Arabic:** Handle right-to-left text correctly
- **Chinese/Japanese:** Handle multi-character words
- **Any script:** Preserve script-specific formatting

### Script-Specific Patterns

**Devanagari (Hindi):**
- Question patterns:
  - Spelling: `"[word]" की वर्तनी क्या है?`
  - Letter position: `"[word]" का [position] अक्षर क्या है?`
  - Letter count: `"[word]" में कितने अक्षर हैं?`
  - Letter listing: `"[word]" के अक्षर क्या हैं?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `क, म, ल`
  - Letter position: Single character: `क`
  - Letter count: Numeric: `3`
- Examples:
  - `"कमल" की वर्तनी क्या है? क, म, ल।`
  - `"घर" की वर्तनी क्या है? घ, र।`
  - `"पानी" की वर्तनी क्या है? प, आ, न, ी।`
  - `"सूरज" की वर्तनी क्या है? स, ू, र, ज।`
  - `"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।`
  - `"कमल" का पहला अक्षर क्या है? क।`
  - `"कमल" का दूसरा अक्षर क्या है? म।`
  - `"पानी" में कितने अक्षर हैं? 4।`
  - `"घर" में कितने अक्षर हैं? 2।`

**Arabic:**
- Handle right-to-left text correctly
- Preserve diacritics and combining characters
- Maintain proper text direction

**Chinese/Japanese:**
- Handle multi-character words
- Preserve character boundaries
- Maintain proper spacing

### Language-Agnostic Principles

1. **Consistent Structure:** Same logical structure regardless of language
2. **Quote Target Words:** Always quote target words in questions
3. **Unquoted Answers:** Answers are never quoted
4. **Preserve Script:** Maintain script-specific formatting
5. **Handle Unicode:** Properly handle all Unicode characters

---

## Implementation Patterns

### Pattern Detection for Word Quoting

**Question Patterns to Detect:**

1. **Spelling Patterns:**
   - `What is the spelling of [word]?`
   - `What's the spelling of [word]?`
   - `Can you spell [word]?`
   - `Tell me the spelling of [word]`
   - `Write the spelling of [word]?`
   - `Show me the spelling of [word]?`
   - `Provide the spelling of [word]?`
   - `Give me the spelling of [word]?`
   - `Spell [word]?`
   - `Spell out [word]?`
   - `How do you spell [word]?`
   - `What does [word] spell?`
   - `What is [word] spelled as?`
   - `Break down [word] into letters?`

2. **Letter Position Patterns:**
   - `What is the [position] letter in [word]?`
   - `Tell me the [position] letter of [word]?`
   - `Give me the [position] letter of [word]?`
   - `Find the [position] letter of [word]?`
   - `Identify the [position] letter of [word]?`
   - `Can you say the [position] letter in [word]?`
   - `Which letter is at position [N] in [word]?`

3. **Letter Count Patterns:**
   - `How many letters are in [word]?`
   - `How many letters does [word] have?`
   - `Count the number of letters in [word]?`
   - `Count the letters in [word]?`
   - `Find the number of letters in [word]?`
   - `Tell me the number of letters in [word]?`
   - `What is the total letter count for [word]?`
   - `What's the letter count of [word]?`
   - `Give me the letter count of [word]?`
   - `How long is the word [word]?`
   - `What is the length of word [word]?`
   - `How many alphabets are there in [word]?`
   - `Letter count of [word]?`

4. **Letter Listing Patterns:**
   - `What are the letters in [word]?`
   - `Break down [word] into letters?`

5. **Sound Matching Patterns:**
   - `Tell me which word starts with /sound/: [word1] or [word2]?`
   - `Choose the word with starting sound /sound/: [word1] or [word2]?`
   - `Name the word that starts with /sound/: [word1] or [word2]?`
   - `Pick the word that begins with sound /sound/: [word1] or [word2]?`
   - `Which of these begins with /sound/: [word1] or [word2]?`
   - `Find the word starting with /sound/: [word1] or [word2]?`
   - `What word begins with /sound/, [word1] or [word2]?`
   - `Which word has the initial sound /sound/: [word1] or [word2]?`
   - `Say which word has the /sound/ sound at the start: [word1] or [word2]?`

6. **Language-Specific Patterns:**
   - Hindi Spelling: `[word] की वर्तनी क्या है?`
   - Hindi Letter Position: `[word] का [position] अक्षर क्या है?`
   - Hindi Letter Count: `[word] में कितने अक्षर हैं?`
   - Hindi Letter Listing: `[word] के अक्षर क्या हैं?` or `[word] को अक्षरों में तोड़ें?`
   - Adapt for other languages as needed

**Fix:** Wrap `[word]` in double quotes: `"word"`

### Regex Patterns (Reference)

**Note:** These are reference patterns. Actual implementation may vary based on programming language and requirements.

**Detect unquoted words in spelling questions:**
```regex
(What is|What's|Can you|Tell me|Write|Show me|Provide|Give me|Spell|Spell out|How do you spell|What does|What is.*spelled as|Break down)\s+(?:the\s+)?(?:spelling\s+of\s+|letters\s+in\s+)?([a-zA-Z]+|[^\s]+)
```

**Detect unquoted words in letter position questions:**
```regex
(What is|Tell me|Give me|Find|Identify|Can you say|Which letter is at position)\s+(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+letter\s+(?:in|of)\s+([a-zA-Z]+|[^\s]+)
```

**Detect unquoted words in letter count questions:**
```regex
(How many letters|Count|Find|Tell me|What is|What's|Give me|How long|How many alphabets|Letter count)\s+(?:are\s+in|does|in|of|for|is|are\s+there\s+in)\s+([a-zA-Z]+|[^\s]+)
```

**Important:** These patterns are simplified examples. Real implementation should:
- Handle Unicode characters
- Avoid matching already-quoted words
- Handle language-specific patterns
- Consider context carefully

---

## Edge Cases and Special Scenarios

### 1. Already-Quoted Words

**Problem:** Word is already quoted in the source data.

**Solution:** Skip quoting. Don't add double quotes.

**Example:**
- Input: `What is the spelling of "cat"?`
- Output: `What is the spelling of "cat"?` (no change)

### 2. Words in Answers

**Problem:** Words appear in answer explanations or context.

**Solution:** Don't quote words in answers. Only quote target words in questions.

**Example:**
- Question: `What is the spelling of "cat"?`
- Answer: `c, a, t` (not `"c", "a", "t"`)

### 3. Multiple Words in One Question

**Problem:** Sound matching questions have multiple candidate words.

**Solution:** Quote all candidate words.

**Example:**
```
Tell me which word starts with /ch/: "dog" or "chair"? chair.
```

### 4. Punctuation in Words

**Problem:** Words contain punctuation (hyphens, apostrophes, etc.).

**Solution:** Include punctuation in quoted word, preserve in answer.

**Example:**
```
What is the spelling of "don't"? d, o, n, ', t.
```

### 5. Case Sensitivity

**Problem:** Words may have different cases.

**Solution:** Preserve original case in questions. Answers typically lowercase for common words.

**Example:**
```
What is the spelling of "Apple"? A, p, p, l, e.
```

### 6. Multi-Character Words (Non-English Scripts)

**Problem:** Words in non-English scripts may have multiple characters per "letter".

**Solution:** Preserve script-specific formatting. Handle characters correctly.

**Example (Hindi):**
```
"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
```

### 7. Empty or Missing Words

**Problem:** Source data may have missing or empty words.

**Solution:** Skip processing or handle gracefully. Don't create malformed Q?A pairs.

### 8. Special Characters in Sound Notation

**Problem:** Sound notation uses forward slashes (e.g., `/ch/`, `/p/`).

**Solution:** Preserve sound notation as-is. Don't quote the sound notation.

**Example:**
```
Tell me which word starts with /ch/: "dog" or "chair"? chair.
```

---

## Quality Assurance

### Validation Checklist

After applying formatting fixes, validate:

**Format Structure:**
- [ ] All questions end with `?`
- [ ] All answers end with `.`
- [ ] Q?A pairs are separated by `. ` (period + space)
- [ ] No line breaks between Q?A pairs
- [ ] No extra spaces or punctuation between pairs

**Word Quoting:**
- [ ] All target words in spelling questions are quoted
- [ ] All target words in letter position questions are quoted
- [ ] All target words in letter count questions are quoted
- [ ] All target words in letter listing questions are quoted
- [ ] All candidate words in sound matching questions are quoted
- [ ] No double-quoting of words
- [ ] Answer letters/characters are NOT quoted
- [ ] Answer numbers are NOT quoted
- [ ] Answer words (in sound matching) are NOT quoted

**Language-Specific:**
- [ ] Language-specific characters are preserved correctly
- [ ] Unicode characters (Devanagari, Arabic, etc.) are handled properly
- [ ] Multi-character words and diacritics are not broken
- [ ] Script-specific formatting is maintained

**Answer Format:**
- [ ] Spelling answers are comma-separated letters: `c, a, t`
- [ ] Letter position answers are single letters: `a`
- [ ] Letter count answers are numeric: `3`
- [ ] Sound matching answers are unquoted words: `chair`
- [ ] Language-specific answers follow correct format

### Automated Testing

**Test Cases:**

1. **English Spelling:**
   - Input: `What is the spelling of behavior?`
   - Expected: `What is the spelling of "behavior"?`

2. **Hindi Spelling:**
   - Input: `कमल की वर्तनी क्या है?`
   - Expected: `"कमल" की वर्तनी क्या है?`

3. **Letter Position:**
   - Input: `What is the first letter in apple?`
   - Expected: `What is the first letter in "apple"?`

4. **Letter Count:**
   - Input: `How many letters are in cat?`
   - Expected: `How many letters are in "cat"?`

5. **Sound Matching:**
   - Input: `Tell me which word starts with /ch/: dog or chair?`
   - Expected: `Tell me which word starts with /ch/: "dog" or "chair"?`

6. **Already Quoted:**
   - Input: `What is the spelling of "cat"?`
   - Expected: `What is the spelling of "cat"?` (no change)

7. **Answer Format:**
   - Verify answers are not quoted
   - Verify correct comma spacing in spelling answers
   - Verify numeric format for letter counts

### Manual Review

**Review Sample:**
- Randomly sample 100-200 Q?A pairs from the dataset
- Verify format structure
- Verify word quoting
- Verify answer format
- Check for edge cases

**Review Checklist:**
- [ ] Format structure is correct
- [ ] Word quoting is consistent
- [ ] Answer format is correct
- [ ] Language-specific characters are preserved
- [ ] No double-quoting
- [ ] No missing quotes
- [ ] No formatting errors

---

## Common Mistakes and Fixes

### Mistake 1: Missing Quotes on Target Words

**Problem:**
```
What is the spelling of behavior? b, e, h, a, v, i, o, r.
```

**Fix:**
```
What is the spelling of "behavior"? b, e, h, a, v, i, o, r.
```

### Mistake 2: Quoting Answer Letters

**Problem:**
```
What is the spelling of "cat"? "c", "a", "t".
```

**Fix:**
```
What is the spelling of "cat"? c, a, t.
```

### Mistake 3: Double-Quoting

**Problem:**
```
What is the spelling of ""cat""? c, a, t.
```

**Fix:**
```
What is the spelling of "cat"? c, a, t.
```

### Mistake 4: Missing Quotes in Sound Matching

**Problem:**
```
Tell me which word starts with /ch/: dog or chair? chair.
```

**Fix:**
```
Tell me which word starts with /ch/: "dog" or "chair"? chair.
```

### Mistake 5: Missing Period After Answer

**Problem:**
```
What is the spelling of "cat"? c, a, t What is the spelling of "dog"? d, o, g.
```

**Fix:**
```
What is the spelling of "cat"? c, a, t. What is the spelling of "dog"? d, o, g.
```

### Mistake 6: Missing Space Between Pairs

**Problem:**
```
What is the spelling of "cat"? c, a, t.What is the spelling of "dog"? d, o, g.
```

**Fix:**
```
What is the spelling of "cat"? c, a, t. What is the spelling of "dog"? d, o, g.
```

### Mistake 7: Extra Spaces

**Problem:**
```
What is the spelling of "cat"? c, a, t.  What is the spelling of "dog"? d, o, g.
```

**Fix:**
```
What is the spelling of "cat"? c, a, t. What is the spelling of "dog"? d, o, g.
```

### Mistake 8: Missing Question Mark

**Problem:**
```
What is the spelling of "cat" c, a, t.
```

**Fix:**
```
What is the spelling of "cat"? c, a, t.
```

### Mistake 9: Language-Specific Characters Not Preserved

**Problem:**
```
कमल की वर्तनी क्या है? (missing quotes)
घर की वर्तनी क्या है? (missing quotes)
पानी में कितने अक्षर हैं? (missing quotes)
```

**Fix:**
```
"कमल" की वर्तनी क्या है? क, म, ल।
"घर" की वर्तनी क्या है? घ, र।
"पानी" में कितने अक्षर हैं? 4।
```

### Mistake 10: Incorrect Answer Format for Letter Count

**Problem:**
```
How many letters are in "cat"? "3".
```

**Fix:**
```
How many letters are in "cat"? 3.
```

---

## Summary

### Dataset Format

**Structure:** 
- **English:** `Q? A. Q? A. Q? A. ...`
- **Hindi:** `Q? A। Q? A। Q? A। ...`

**Key Points:**
- Question ends with `?` followed by space
- Answer ends with language-specific punctuation:
  - **English:** `.` (period)
  - **Hindi:** `।` (Devanagari danda)
- Pairs separated by punctuation + space:
  - **English:** `. ` (period + space)
  - **Hindi:** `। ` (danda + space)
- Continuous format (no line breaks between pairs)

### Word Quoting Rule

**Rule:** Wrap target words in double quotes in questions.

**Examples (English):**
- `What is the spelling of word?` → `What is the spelling of "word"?`
- `How many letters are in cat?` → `How many letters are in "cat"?`
- `Tell me which word starts with /ch/: dog or chair?` → `Tell me which word starts with /ch/: "dog" or "chair"?`

**Examples (Hindi):**
- `कमल की वर्तनी क्या है?` → `"कमल" की वर्तनी क्या है?`
- `घर की वर्तनी क्या है?` → `"घर" की वर्तनी क्या है?`
- `पानी में कितने अक्षर हैं?` → `"पानी" में कितने अक्षर हैं?`
- `कमल का पहला अक्षर क्या है?` → `"कमल" का पहला अक्षर क्या है?`
- `विद्यालय की वर्तनी क्या है?` → `"विद्यालय" की वर्तनी क्या है?`

### Answer Format

- **Spelling:** Comma-separated letters: `c, a, t`
- **Letter Position:** Single letter: `a`
- **Letter Count:** Numeric: `3`
- **Sound Matching:** Unquoted word: `chair`
- **Language-Specific:** Follow script-specific format

### Key Principles

1. **Quote target words in questions, not answers**
2. **Preserve language-specific characters and script formatting**
3. **Maintain consistent format structure**
4. **Handle Unicode correctly**
5. **Apply rules uniformly across entire dataset**

### Scope

This guide applies to all Group 1 (Language and Literacy) questions involving target words, regardless of language or script.

---

**Last Updated:** Based on `output/group1.txt` dataset analysis
**Version:** 2.0 (Comprehensive)
