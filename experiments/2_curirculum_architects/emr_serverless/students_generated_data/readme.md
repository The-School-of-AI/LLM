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
  - **Indic Languages (Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Assamese, Kannada, Malayalam, Odia, Punjabi):** Devanagari danda (`।`) - `Q? A। Q? A। ...`
- Each `Q?A` pair is separated by punctuation + space:
  - **English:** `. ` (period + space)
  - **Indic Languages:** `। ` (danda + space)
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
     - **Indic Languages (Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Assamese, Kannada, Malayalam, Odia, Punjabi):** Devanagari danda (`।`)
   - For spelling questions: comma-separated letters (e.g., `c, a, t`)
   - For letter position questions: single letter (e.g., `c`)
   - For letter count questions: numeric value (e.g., `3`)
   - For sound/choice questions: the selected word (e.g., `chair`)

3. **Pair Separation:**
   - **English:** Exactly one period (`.`) followed by exactly one space (` `)
   - **Indic Languages:** Exactly one Devanagari danda (`।`) followed by exactly one space (` `)
   - No additional punctuation between pairs
   - No line breaks or newlines

### Examples from `group1.txt`

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

**✅ Correct Format (Indic Languages):**

Format: `Q? A। Q? A। Q? A। ...` (Note: All Indic languages use Devanagari danda `।` instead of period `.`)

**Hindi (हिंदी):**
```
"कमल" की वर्तनी क्या है? क, म, ल। "घर" की वर्तनी क्या है? घ, र। "पानी" की वर्तनी क्या है? प, आ, न, ी। "सूरज" की वर्तनी क्या है? स, ू, र, ज। "विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
```

**Bengali (বাংলা):**
```
"বই" এর বানান কী? ব, ই। "ঘর" এর বানান কী? ঘ, র। "জল" এর বানান কী? জ, ল। "সূর্য" এর বানান কী? স, ূ, র, ্, য। "বিদ্যালয়" এর বানান কী? ব, ি, দ, ্, য, া, ল, য়।
```

**Tamil (தமிழ்):**
```
"புத்தகம்" எப்படி எழுதுவது? ப, ு, த, ், த, க, ம, ்। "வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு। "நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்। "சூரியன்" எப்படி எழுதுவது? ச, ூ, ர, ி, ய, ன, ்।
```

**Telugu (తెలుగు):**
```
"పుస్తకం" స్పెల్లింగ్ ఏమిటి? ప, ు, స, ్, త, క, ం। "ఇల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు। "నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు।
```

**Gujarati (ગુજરાતી):**
```
"પુસ્તક" ની જોડણી શું છે? પ, ુ, સ, ્, ત, ક। "ઘર" ની જોડણી શું છે? ઘ, ર। "પાણી" ની જોડણી શું છે? પ, ા, ણ, ી।
```

**Marathi (मराठी):**
```
"पुस्तक" ची शुद्धलेखन काय आहे? प, ु, स, ्, त, क। "घर" ची शुद्धलेखन काय आहे? घ, र। "पाणी" ची शुद्धलेखन काय आहे? प, ा, ण, ी।
```

**Assamese (অসমীয়া):**
```
"কিতাপ" শব্দের বানান কি? ক, ি, ত, া, প। "ঘৰ" শব্দের বানান কি? ঘ, ৰ। "পানী" শব্দের বানান কি? প, া, ন, ী।
```

**Kannada (ಕನ್ನಡ):**
```
"ಪುಸ್ತಕ" ಕ್ಕೆ ಸ್ಪೆಲಿಂಗ್ ಏನು? ಪ, ು, ಸ, ್, ತ, ಕ। "ಮನೆ" ಕ್ಕೆ ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ। "ನೀರು" ಕ್ಕೆ ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।
```

**Malayalam (മലയാളം):**
```
"പുസ്തകം" എങ്ങനെ എഴുതുന്നു? പ, ു, സ, ്, ത, ക, ം। "വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്। "നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।
```

**Odia (ଓଡ଼ିଆ):**
```
"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି। "ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର। "ପାଣି" ଶବ୍ଦର ବାନାନ କଣ? ପ, ା, ଣ, ି।
```

**Punjabi (ਪੰਜਾਬੀ):**
```
"ਕਿਤਾਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਕ, ਿ, ਤ, ਾ, ਬ। "ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ। "ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।
```

**❌ Incorrect Format:**

- **Missing punctuation after answer:** `What is the spelling of "behavior"? b, e, h, a, v, i, o, r What's the spelling of "curry"?`
- **Missing question mark:** `What is the spelling of "behavior" b, e, h, a, v, i, o, r.`
- **Missing space after question mark:** `What is the spelling of "behavior"?b, e, h, a, v, i, o, r.` (should be `? `)
- **Missing space between pairs:** `What is the spelling of "behavior"? b, e, h, a, v, i, o, r.What's the spelling of "curry"?`
- **Wrong punctuation for Indic languages:** Using period (`.`) instead of Devanagari danda (`।`) for Hindi/Bengali/Tamil/Telugu/Gujarati/Marathi
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

**Examples (Bengali):**
```
"বই" এর বানান কী? ব, ই। "ঘর" এর বানান কী? ঘ, র। "জল" এর বানান কী? জ, ল।
```

**Examples (Tamil):**
```
"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு। "நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்। "பள்ளி" எப்படி எழுதுவது? ப, ள, ், ள, ி।
```

**Examples (Telugu):**
```
"ఇల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు। "నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు। "పువ్వు" స్పెల్లింగ్ ఏమిటి? ప, ు, వ, ్, వ, ు।
```

**Examples (Gujarati):**
```
"ઘર" ની જોડણી શું છે? ઘ, ર। "પાણી" ની જોડણી શું છે? પ, ા, ણ, ી। "શાળા" ની જોડણી શું છે? શ, ા, ળ, ા।
```

**Examples (Marathi):**
```
"घर" ची शुद्धलेखन काय आहे? घ, र। "पाणी" ची शुद्धलेखन काय आहे? प, ा, ण, ी। "फूल" ची शुद्धलेखन काय आहे? फ, ू, ल।
```

**Examples (Assamese):**
```
"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প। "ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ। "পানী" শব্দেৰ বানান কি? প, া, ন, ী।
```

**Examples (Kannada):**
```
"ಪುಸ್ತಕ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಪ, ು, ಸ, ್, ತ, ಕ। "ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ। "ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।
```

**Examples (Malayalam):**
```
"പുസ്തകം" എങ്ങനെ എഴുതുന്നു? പ, ു, സ, ്, ത, ക, ം। "വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്। "നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।
```

**Examples (Odia):**
```
"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି। "ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର। "ପାଣି" ଶବ୍ଦର ବାନାନ କଣ? ପ, ା, ଣ, ି।
```

**Examples (Punjabi):**
```
"ਕਿਤਾਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਕ, ਿ, ਤ, ਾ, ਬ। "ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ। "ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।
```

**Answer Format:** Comma-separated letters: `b, e, h, a, v, i, o, r` (English), `क, म, ल` (Hindi), `ব, ই` (Bengali), `வ, ீ, ட, ு` (Tamil), `ఇ, ల, ్, ల, ు` (Telugu), `ઘ, ર` (Gujarati), `घ, र` (Marathi), `ক, ি, ত, া, প` (Assamese), `ಮ, ನ, ೆ` (Kannada), `വ, ീ, ട, ്` (Malayalam), `ବ, ହ, ି` (Odia), `ਘ, ਰ` (Punjabi)

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

**Examples (Bengali):**
```
"বই" এর প্রথম অক্ষর কী? ব। "জল" এর দ্বিতীয় অক্ষর কী? ল। "ঘর" এর প্রথম অক্ষর কী? ঘ।
```

**Examples (Tamil):**
```
"வீடு" இன் முதல் எழுத்து என்ன? வ। "நீர்" இன் இரண்டாவது எழுத்து என்ன? ீ। "பள்ளி" இன் முதல் எழுத்து என்ன? ப।
```

**Examples (Telugu):**
```
"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి? ఇ। "నీరు" యొక్క రెండవ అక్షరం ఏమిటి? ీ। "పువ్వు" యొక్క మొదటి అక్షరం ఏమిటి? ప।
```

**Examples (Gujarati):**
```
"ઘર" નો પ્રથમ અક્ષર શું છે? ઘ। "પાણી" નો બીજો અક્ષર શું છે? ા। "શાળા" નો પ્રથમ અક્ષર શું છે? શ।
```

**Examples (Marathi):**
```
"घर" चे पहिले अक्षर काय आहे? घ। "पाणी" चे दुसरे अक्षर काय आहे? ा। "फूल" चे पहिले अक्षर काय आहे? फ।
```

**Examples (Assamese):**
```
"কিতাপ" শব্দৰ প্ৰথম আখৰ কি? ক। "ঘৰ" শব্দৰ দ্বিতীয় আখৰ কি? ৰ। "পানী" শব্দৰ প্ৰথম আখৰ কি? প।
```

**Examples (Kannada):**
```
"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಮ। "ನೀರು" ಯ ಎರಡನೇ ಅಕ್ಷರ ಏನು? ೀ। "ಪುಸ್ತಕ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಪ।
```

**Examples (Malayalam):**
```
"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? വ। "നീര്" ന്റെ രണ്ടാമത്തെ അക്ഷരം എന്താണ്? ീ। "പുസ്തകം" ന്റെ ആദ്യ അക്ഷരം എന്താണ്? പ।
```

**Examples (Odia):**
```
"ବହି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ବ। "ଘର" ର ଦ୍ୱିତୀୟ ଅକ୍ଷର କଣ? ର। "ପାଣି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ପ।
```

**Examples (Punjabi):**
```
"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਘ। "ਪਾਣੀ" ਦਾ ਦੂਜਾ ਅੱਖਰ ਕੀ ਹੈ? ਾ। "ਕਿਤਾਬ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਕ।
```

**Answer Format:** Single letter: `a`, `p`, `p`, `l` (English), `क`, `म`, `ल` (Hindi), `ব`, `ল` (Bengali), `வ`, `ீ` (Tamil), `ఇ`, `ీ` (Telugu), `ઘ`, `ા` (Gujarati), `घ`, `ा` (Marathi), `ক`, `ৰ` (Assamese), `ಮ`, `ೀ` (Kannada), `വ`, `ീ` (Malayalam), `ବ`, `ର` (Odia), `ਘ`, `ਾ` (Punjabi)


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

**Examples (Bengali):**
```
"বই" এ কতটি অক্ষর আছে? 2। "ঘর" এ কতটি অক্ষর আছে? 2। "জল" এ কতটি অক্ষর আছে? 2। "বিদ্যালয়" এ কতটি অক্ষর আছে? 7।
```

**Examples (Tamil):**
```
"வீடு" இல் எத்தனை எழுத்துக்கள்? 4। "நீர்" இல் எத்தனை எழுத்துக்கள்? 4। "பள்ளி" இல் எத்தனை எழுத்துக்கள்? 5।
```

**Examples (Telugu):**
```
"ఇల్లు" లో ఎన్ని అక్షరాలు? 5। "నీరు" లో ఎన్ని అక్షరాలు? 4। "పువ్వు" లో ఎన్ని అక్షరాలు? 5।
```

**Examples (Gujarati):**
```
"ઘર" માં કેટલા અક્ષર છે? 2। "પાણી" માં કેટલા અક્ષર છે? 4। "શાળા" માં કેટલા અક્ષર છે? 4।
```

**Examples (Marathi):**
```
"घर" मध्ये किती अक्षरे आहेत? 2। "पाणी" मध्ये किती अक्षरे आहेत? 4। "फूल" मध्ये किती अक्षरे आहेत? 3।
```

**Examples (Assamese):**
```
"কিতাপ" ত কিমান আখৰ আছে? 5। "ঘৰ" ত কিমান আখৰ আছে? 2। "পানী" ত কিমান আখৰ আছে? 4।
```

**Examples (Kannada):**
```
"ಮನೆ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 3। "ನೀರು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4। "ಪುಸ್ತಕ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 6।
```

**Examples (Malayalam):**
```
"വീട്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4। "നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4। "പുസ്തകം" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 7।
```

**Examples (Odia):**
```
"ବହି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 3। "ଘର" ରେ କେତେ ଅକ୍ଷର ଅଛି? 2। "ପାଣି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 4।
```

**Examples (Punjabi):**
```
"ਘਰ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 2। "ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4। "ਕਿਤਾਬ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 5।
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

**Examples (Bengali):**
```
"বই" এর অক্ষরগুলি কী? ব, ই। "ঘর" কে অক্ষরে ভাগ করুন? ঘ, র। "জল" এর অক্ষরগুলি কী? জ, ল।
```

**Examples (Tamil):**
```
"வீடு" வை எழுத்துக்களாகப் பிரிக்கவும்? வ, ீ, ட, ு। "நீர்" இன் எழுத்துக்கள் என்ன? ந, ீ, ர, ்।
```

**Examples (Telugu):**
```
"ఇల్లు" ను అక్షరాలుగా విభజించండి? ఇ, ల, ్, ల, ు। "నీరు" యొక్క అక్షరాలు ఏమిటి? న, ీ, ర, ు।
```

**Examples (Gujarati):**
```
"ઘર" ને અક્ષરોમાં વિભાજિત કરો? ઘ, ર। "પાણી" ના અક્ષરો શું છે? પ, ા, ણ, ી।
```

**Examples (Marathi):**
```
"घर" ची अक्षरे काय आहेत? घ, र। "पाणी" ला अक्षरांमध्ये विभाजित करा? प, ा, ण, ी।
```

**Examples (Assamese):**
```
"কিতাপ" ৰ আখৰবোৰ কি? ক, ি, ত, া, প। "ঘৰ" ক আখৰত ভাগ কৰক? ঘ, ৰ। "পানী" ৰ আখৰবোৰ কি? প, া, ন, ী।
```

**Examples (Kannada):**
```
"ಮನೆ" ಯ ಅಕ್ಷರಗಳು ಯಾವುವು? ಮ, ನ, ೆ। "ನೀರು" ಅಕ್ಷರಗಳಾಗಿ ಬೇರ್ಪಡಿಸಿ? ನ, ೀ, ರ, ು।
```

**Examples (Malayalam):**
```
"വീട്" ലെ അക്ഷരങ്ങൾ എന്തെല്ലാമാണ്? വ, ീ, ട, ്। "നീര്" അക്ഷരങ്ങളായി വേർതിരിക്കുക? ന, ീ, ര, ്।
```

**Examples (Odia):**
```
"ବହି" ର ଅକ୍ଷର ଗୁଡିକ କଣ? ବ, ହ, ି। "ଘର" କୁ ଅକ୍ଷରରେ ବିଭକ୍ତ କରନ୍ତୁ? ଘ, ର।
```

**Examples (Punjabi):**
```
"ਘਰ" ਦੇ ਅੱਖਰ ਕੀ ਹਨ? ਘ, ਰ। "ਪਾਣੀ" ਨੂੰ ਅੱਖਰਾਂ ਵਿੱਚ ਵੰਡੋ? ਪ, ਾ, ਣ, ੀ।
```

**Answer Format:** Comma-separated letters: `f, i, l, e` (English), `क, म, ल` (Hindi), `ব, ই` (Bengali), `வ, ீ, ட, ு` (Tamil), `ఇ, ల, ్, ల, ు` (Telugu), `ઘ, ર` (Gujarati), `घ, र` (Marathi), `ক, ি, ত, া, প` (Assamese), `ಮ, ನ, ೆ` (Kannada), `വ, ീ, ട, ്` (Malayalam), `ବ, ହ, ି` (Odia), `ਘ, ਰ` (Punjabi)


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

**Examples (English):**
```
Tell me which word starts with /ch/: "dog" or "chair"? chair. Choose the word with starting sound /p/: "blame" or "party"? party. Which word has the initial sound /sm/: "dog" or "smell"? smell.
```

**Examples (Hindi):**
```
कौन सा शब्द 'क' से शुरू होता है: "घर" या "कमल"? कमल। पहले अक्षर 'प' वाला शब्द बताओ: "पानी" या "घर"? पानी।
```

**Examples (Bengali):**
```
কোন শব্দটি 'ব' দিয়ে শুরু হয়: "ঘর" বা "বই"? বই। 'জ' অক্ষর দিয়ে কোন শব্দ শুরু হয়: "ফুল" বা "জল"? জল।
```

**Examples (Tamil):**
```
எந்த சொல் 'வ' என்று தொடங்குகிறது: "நீர்" அல்லது "வீடு"? வீடு। 'ம' எழுத்துடன் தொடங்கும் சொல்: "மலர்" அல்லது "நீர்"? மலர்।
```

**Examples (Telugu):**
```
ఏ పదం 'ఇ' తో ప్రారంభం అవుతుంది: "నీరు" లేదా "ఇల్లు"? ఇల్లు। 'ప' అక్షరంతో ప్రారంభమైన పదం: "నీరు" లేదా "పువ్వు"? పువ్వు।
```

**Examples (Gujarati):**
```
કોણું શબ્દ 'ઘ' થી શરૂ થાય છે: "પાણી" કે "ઘર"? ઘર। 'શ' અક્ષરથી કોણું શબ્દ શરૂ થાય છે: "ફૂલ" કે "શાળા"? શાળા।
```

**Examples (Marathi):**
```
कोणता शब्द 'प' ने सुरू होतो: "घर" किंवा "पाणी"? पाणी। 'फ' अक्षराने कोणता शब्द सुरू होतो: "घर" किंवा "फूल"? फूल।
```

**Examples (Assamese):**
```
কোন শব্দ 'ক' ৰে আৰম্ভ হয়: "ঘৰ" নে "কিতাপ"? কিতাপ। 'প' আখৰেৰে কোন শব্দ আৰম্ভ হয়: "ঘৰ" নে "পানী"? পানী।
```

**Examples (Kannada):**
```
ಯಾವ ಪದ 'ಮ' ನಿಂದ ಪ್ರಾರಂಭವಾಗುತ್ತದೆ: "ನೀರು" ಅಥವಾ "ಮನೆ"? ಮನೆ। 'ಪ' ಅಕ್ಷರದಿಂದ ಯಾವ ಪದ ಪ್ರಾರಂಭವಾಗುತ್ತದೆ: "ನೀರು" ಅಥವಾ "ಪುಸ್ತಕ"? ಪುಸ್ತಕ।
```

**Examples (Malayalam):**
```
ഏത് വാക്ക് 'വ' കൊണ്ട് ആരംഭിക്കുന്നു: "നീര്" അല്ലെങ്കിൽ "വീട്"? വീട്। 'പ' അക്ഷരം കൊണ്ട് ഏത് വാക്ക് തുടങ്ങുന്നു: "നീര്" അല്ലെങ്കിൽ "പുസ്തകം"? പുസ്തകം।
```

**Examples (Odia):**
```
କେଉଁ ଶବ୍ଦ 'ବ' ରୁ ଆରମ୍ଭ ହୁଏ: "ଘର" କିମ୍ବା "ବହି"? ବହି। 'ପ' ଅକ୍ଷରରୁ କେଉଁ ଶବ୍ଦ ଆରମ୍ଭ ହୁଏ: "ଘର" କିମ୍ବା "ପାଣି"? ପାଣି।
```

**Examples (Punjabi):**
```
ਕਿਹੜਾ ਸ਼ਬਦ 'ਘ' ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ: "ਪਾਣੀ" ਜਾਂ "ਘਰ"? ਘਰ। 'ਕ' ਅੱਖਰ ਨਾਲ ਕਿਹੜਾ ਸ਼ਬਦ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ: "ਪਾਣੀ" ਜਾਂ "ਕਿਤਾਬ"? ਕਿਤਾਬ।
```

**Answer Format:** 
- **English:** The selected word: `chair`, `party`, `smell`
- **Indic Languages:** The selected word: `कमल`, `पानी` (Hindi), `বই`, `জল` (Bengali), `வீடு`, `மலர்` (Tamil), `ఇల్లు`, `పువ్వు` (Telugu), `ઘર`, `શાળા` (Gujarati), `पाणी`, `फूल` (Marathi), `কিতাপ`, `পানী` (Assamese), `ಮನೆ`, `ಪುಸ್ತಕ` (Kannada), `വീട്`, `പുസ്തകം` (Malayalam), `ବହି`, `ପାଣି` (Odia), `ਘਰ`, `ਕਿਤਾਬ` (Punjabi)


**Note:** Sound notation uses forward slashes (e.g., `/ch/`, `/p/`, `/th/`)

### 6. Language-Specific Question Patterns

Different languages may have different question patterns. Here are examples:

**Hindi (हिंदी) - Devanagari Script:**

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

**Bengali (বাংলা) - Bengali Script:**

**Spelling Questions:**
- `"বই" এর বানান কী?` (What is the spelling of "boi"?)
- `"ঘর" এর বানান কী?` (What is the spelling of "ghor"?)
- `"জল" এর বানান কী?` (What is the spelling of "jol"?)
- `"সূর্য" এর বানান কী?` (What is the spelling of "surjo"?)
- `"বিদ্যালয়" এর বানান কী?` (What is the spelling of "bidyaloy"?)
- `"ফুল" এর বানান কী?` (What is the spelling of "phul"?)
- `"গাছ" এর বানান কী?` (What is the spelling of "gachh"?)
- `"শিশু" এর বানান কী?` (What is the spelling of "shishu"?)
- `"স্কুল" এর বানান কী?` (What is the spelling of "school"?)

**Letter Position Questions:**
- `"বই" এর প্রথম অক্ষর কী?` (What is the first letter of "boi"?)
- `"জল" এর দ্বিতীয় অক্ষর কী?` (What is the second letter of "jol"?)

**Letter Count Questions:**
- `"বই" এ কতটি অক্ষর আছে?` (How many letters are in "boi"?)
- `"ঘর" এ কতটি অক্ষর আছে?` (How many letters are in "ghor"?)
- `"জল" এ কতটি অক্ষর আছে?` (How many letters are in "jol"?)

**Letter Listing Questions:**
- `"বই" এর অক্ষরগুলি কী?` (What are the letters in "boi"?)
- `"ঘর" কে অক্ষরে ভাগ করুন?` (Break down "ghor" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ব, ই` or `ঘ, র` or `জ, ল`
- Letter Position: Single character: `ব`, `র`
- Letter Count: Numeric value: `2`, `2`, `2`

---

**Tamil (தமிழ்) - Tamil Script:**

**Spelling Questions:**
- `"புத்தகம்" எப்படி எழுதுவது?` (How to write "puththagam"?)
- `"வீடு" எப்படி எழுதுவது?` (How to write "veedu"?)
- `"நீர்" எப்படி எழுதுவது?` (How to write "neer"?)
- `"சூரியன்" எப்படி எழுதுவது?` (How to write "suriyan"?)
- `"பள்ளி" எப்படி எழுதுவது?` (How to write "palli"?)
- `"மலர்" எப்படி எழுதுவது?` (How to write "malar"?)
- `"மரம்" எப்படி எழுதுவது?` (How to write "maram"?)
- `"குழந்தை" எப்படி எழுதுவது?` (How to write "kuzhandhai"?)

**Letter Position Questions:**
- `"வீடு" இன் முதல் எழுத்து என்ன?` (What is the first letter of "veedu"?)
- `"நீர்" இன் இரண்டாவது எழுத்து என்ன?` (What is the second letter of "neer"?)

**Letter Count Questions:**
- `"வீடு" இல் எத்தனை எழுத்துக்கள்?` (How many letters in "veedu"?)
- `"நீர்" இல் எத்தனை எழுத்துக்கள்?` (How many letters in "neer"?)
- `"மலர்" இல் எத்தனை எழுத்துக்கள்?` (How many letters in "malar"?)

**Letter Listing Questions:**
- `"வீடு" வை எழுத்துக்களாகப் பிரிக்கவும்?` (Break "veedu" into letters?)
- `"மலர்" இன் எழுத்துக்கள் என்ன?` (What are the letters in "malar"?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `வ, ீ, ட, ு` or `ந, ீ, ர, ்`
- Letter Position: Single character: `வ`, `ந`
- Letter Count: Numeric value: `4`, `4`, `4`

---

**Telugu (తెలుగు) - Telugu Script:**

**Spelling Questions:**
- `"పుస్తకం" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "pusthakam"?)
- `"ఇల్లు" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "illu"?)
- `"నీరు" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "neeru"?)
- `"సూర్యుడు" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "suryudu"?)
- `"పాఠశాల" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "paatashaala"?)
- `"పువ్వు" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "puvvu"?)
- `"చెట్టు" స్పెల్లింగ్ ఏమిటి?` (What is the spelling of "chettu"?)

**Letter Position Questions:**
- `"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి?` (What is the first letter of "illu"?)
- `"నీరు" యొక్క రెండవ అక్షరం ఏమిటి?` (What is the second letter of "neeru"?)

**Letter Count Questions:**
- `"ఇల్లు" లో ఎన్ని అక్షరాలు?` (How many letters in "illu"?)
- `"నీరు" లో ఎన్ని అక్షరాలు?` (How many letters in "neeru"?)
- `"పువ్వు" లో ఎన్ని అక్షరాలు?` (How many letters in "puvvu"?)

**Letter Listing Questions:**
- `"ఇల్లు" ను అక్షరాలుగా విభజించండి?` (Break "illu" into letters?)
- `"నీరు" యొక్క అక్షరాలు ఏమిటి?` (What are the letters in "neeru"?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ఇ, ల, ్, ల, ు` or `న, ీ, ర, ు`
- Letter Position: Single character: `ఇ`, `న`
- Letter Count: Numeric value: `5`, `4`, `5`

---

**Gujarati (ગુજરાતી) - Gujarati Script:**

**Spelling Questions:**
- `"પુસ્તક" ની જોડણી શું છે?` (What is the spelling of "pustak"?)
- `"ઘર" ની જોડણી શું છે?` (What is the spelling of "ghar"?)
- `"પાણી" ની જોડણી શું છે?` (What is the spelling of "paani"?)
- `"સૂરજ" ની જોડણી શું છે?` (What is the spelling of "suraj"?)
- `"શાળા" ની જોડણી શું છે?` (What is the spelling of "shaala"?)
- `"ફૂલ" ની જોડણી શું છે?` (What is the spelling of "phool"?)
- `"ઝાડ" ની જોડણી શું છે?` (What is the spelling of "zaad"?)

**Letter Position Questions:**
- `"ઘર" નો પ્રથમ અક્ષર શું છે?` (What is the first letter of "ghar"?)
- `"પાણી" નો બીજો અક્ષર શું છે?` (What is the second letter of "paani"?)

**Letter Count Questions:**
- `"ઘર" માં કેટલા અક્ષર છે?` (How many letters in "ghar"?)
- `"પાણી" માં કેટલા અક્ષર છે?` (How many letters in "paani"?)
- `"સૂરજ" માં કેટલા અક્ષર છે?` (How many letters in "suraj"?)

**Letter Listing Questions:**
- `"ઘર" ને અક્ષરોમાં વિભાજિત કરો?` (Break "ghar" into letters?)
- `"પાણી" ના અક્ષરો શું છે?` (What are the letters in "paani"?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ઘ, ર` or `પ, ા, ણ, ી`
- Letter Position: Single character: `ઘ`, `પ`
- Letter Count: Numeric value: `2`, `4`, `4`

---

**Marathi (मराठी) - Devanagari Script:**

**Spelling Questions:**
- `"पुस्तक" ची शुद्धलेखन काय आहे?` (What is the spelling of "pustak"?)
- `"घर" ची शुद्धलेखन काय आहे?` (What is the spelling of "ghar"?)
- `"पाणी" ची शुद्धलेखन काय आहे?` (What is the spelling of "paani"?)
- `"सूर्य" ची शुद्धलेखन काय आहे?` (What is the spelling of "surya"?)
- `"शाळा" ची शुद्धलेखन काय आहे?` (What is the spelling of "shaala"?)
- `"फूल" ची शुद्धलेखन काय आहे?` (What is the spelling of "phool"?)
- `"झाड" ची शुद्धलेखन काय आहे?` (What is the spelling of "zaad"?)

**Letter Position Questions:**
- `"घर" चे पहिले अक्षर काय आहे?` (What is the first letter of "ghar"?)
- `"पाणी" चे दुसरे अक्षर काय आहे?` (What is the second letter of "paani"?)

**Letter Count Questions:**
- `"घर" मध्ये किती अक्षरे आहेत?` (How many letters in "ghar"?)
- `"पाणी" मध्ये किती अक्षरे आहेत?` (How many letters in "paani"?)
- `"सूर्य" मध्ये किती अक्षरे आहेत?` (How many letters in "surya"?)

**Letter Listing Questions:**
- `"घर" ची अक्षरे काय आहेत?` (What are the letters in "ghar"?)
- `"पाणी" ला अक्षरांमध्ये विभाजित करा?` (Break "paani" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `घ, र` or `प, ा, ण, ी`
- Letter Position: Single character: `घ`, `प`
- Letter Count: Numeric value: `2`, `4`, `4`

---

**Assamese (অসমীয়া) - Assamese Script:**

**Spelling Questions:**
- `"কিতাপ" শব্দেৰ বানান কি?` (What is the spelling of "kitap"?)
- `"ঘৰ" শব্দেৰ বানান কি?` (What is the spelling of "ghor"?)
- `"পানী" শব্দেৰ বানান কি?` (What is the spelling of "paani"?)
- `"সূৰ্য" শব্দেৰ বানান কি?` (What is the spelling of "surjo"?)
- `"বিদ্যালয়" শব্দেৰ বানান কি?` (What is the spelling of "bidyaloy"?)

**Letter Position Questions:**
- `"কিতাপ" ৰ প্ৰথম আখৰ কি?` (What is the first letter of "kitap"?)
- `"ঘৰ" ৰ দ্বিতীয় আখৰ কি?` (What is the second letter of "ghor"?)

**Letter Count Questions:**
- `"কিতাপ" ত কিমান আখৰ আছে?` (How many letters in "kitap"?)
- `"ঘৰ" ত কিমান আখৰ আছে?` (How many letters in "ghor"?)
- `"পানী" ত কিমান আখৰ আছে?` (How many letters in "paani"?)

**Letter Listing Questions:**
- `"কিতাপ" ৰ আখৰবোৰ কি?` (What are the letters in "kitap"?)
- `"ঘৰ" ক আখৰত ভাগ কৰক?` (Break "ghor" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ক, ি, ত, া, প` or `ঘ, ৰ`
- Letter Position: Single character: `ক`, `ৰ`
- Letter Count: Numeric value: `5`, `2`, `4`

---

**Kannada (ಕನ್ನಡ) - Kannada Script:**

**Spelling Questions:**
- `"ಪುಸ್ತಕ" ಸ್ಪೆಲಿಂಗ್ ಏನು?` (What is the spelling of "pusthaka"?)
- `"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು?` (What is the spelling of "mane"?)
- `"ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು?` (What is the spelling of "neeru"?)
- `"ಸೂರ್ಯ" ಸ್ಪೆಲಿಂಗ್ ಏನು?` (What is the spelling of "soorya"?)
- `"ಶಾಲೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು?` (What is the spelling of "shaale"?)

**Letter Position Questions:**
- `"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು?` (What is the first letter of "mane"?)
- `"ನೀರು" ಯ ಎರಡನೇ ಅಕ್ಷರ ಏನು?` (What is the second letter of "neeru"?)

**Letter Count Questions:**
- `"ಮನೆ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?` (How many letters in "mane"?)
- `"ನೀರು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?` (How many letters in "neeru"?)
- `"ಪುಸ್ತಕ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?` (How many letters in "pusthaka"?)

**Letter Listing Questions:**
- `"ಮನೆ" ಯ ಅಕ್ಷರಗಳು ಯಾವುವು?` (What are the letters in "mane"?)
- `"ನೀರು" ಅಕ್ಷರಗಳಾಗಿ ಬೇರ್ಪಡಿಸಿ?` (Break "neeru" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ಮ, ನ, ೆ` or `ನ, ೀ, ರ, ು`
- Letter Position: Single character: `ಮ`, `ೀ`
- Letter Count: Numeric value: `3`, `4`, `6`

---

**Malayalam (മലയാളം) - Malayalam Script:**

**Spelling Questions:**
- `"പുസ്തകം" എങ്ങനെ എഴുതുന്നു?` (How to write "pusthakam"?)
- `"വീട്" എങ്ങനെ എഴുതുന്നു?` (How to write "veedu"?)
- `"നീര്" എങ്ങനെ എഴുതുന്നു?` (How to write "neer"?)
- `"സൂര്യന്" എങ്ങനെ എഴുതുന്നു?` (How to write "sooryan"?)
- `"പാഠശാല" എങ്ങനെ എഴുതുന്നു?` (How to write "paatashaala"?)

**Letter Position Questions:**
- `"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്?` (What is the first letter of "veedu"?)
- `"നീര്" ന്റെ രണ്ടാമത്തെ അക്ഷരം എന്താണ്?` (What is the second letter of "neer"?)

**Letter Count Questions:**
- `"വീട്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്?` (How many letters in "veedu"?)
- `"നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്?` (How many letters in "neer"?)
- `"പുസ്തകം" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്?` (How many letters in "pusthakam"?)

**Letter Listing Questions:**
- `"വീട്" ലെ അക്ഷരങ്ങൾ എന്തെല്ലാമാണ്?` (What are the letters in "veedu"?)
- `"നീര്" അക്ഷരങ്ങളായി വേർതിരിക്കുക?` (Break "neer" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `വ, ീ, ട, ്` or `ന, ീ, ര, ്`
- Letter Position: Single character: `വ`, `ീ`
- Letter Count: Numeric value: `4`, `4`, `7`

---

**Odia (ଓଡ଼ିଆ) - Odia Script:**

**Spelling Questions:**
- `"ବହି" ଶବ୍ଦର ବାନାନ କଣ?` (What is the spelling of "bahi"?)
- `"ଘର" ଶବ୍ଦର ବାନାନ କଣ?` (What is the spelling of "ghara"?)
- `"ପାଣି" ଶବ୍ଦର ବାନାନ କଣ?` (What is the spelling of "paani"?)
- `"ସୂର୍ଯ୍ୟ" ଶବ୍ଦର ବାନାନ କଣ?` (What is the spelling of "surya"?)
- `"ବିଦ୍ୟାଳୟ" ଶବ୍ଦର ବାନାନ କଣ?` (What is the spelling of "bidyalaya"?)

**Letter Position Questions:**
- `"ବହି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ?` (What is the first letter of "bahi"?)
- `"ଘର" ର ଦ୍ୱିତୀୟ ଅକ୍ଷର କଣ?` (What is the second letter of "ghara"?)

**Letter Count Questions:**
- `"ବହି" ରେ କେତେ ଅକ୍ଷର ଅଛି?` (How many letters in "bahi"?)
- `"ଘର" ରେ କେତେ ଅକ୍ଷର ଅଛି?` (How many letters in "ghara"?)
- `"ପାଣି" ରେ କେତେ ଅକ୍ଷର ଅଛି?` (How many letters in "paani"?)

**Letter Listing Questions:**
- `"ବହି" ର ଅକ୍ଷର ଗୁଡିକ କଣ?` (What are the letters in "bahi"?)
- `"ଘର" କୁ ଅକ୍ଷରରେ ବିଭକ୍ତ କରନ୍ତୁ?` (Break "ghara" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ବ, ହ, ି` or `ଘ, ର`
- Letter Position: Single character: `ବ`, `ର`
- Letter Count: Numeric value: `3`, `2`, `4`

---

**Punjabi (ਪੰਜਾਬੀ) - Gurmukhi Script:**

**Spelling Questions:**
- `"ਕਿਤਾਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` (What is the spelling of "kitaab"?)
- `"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` (What is the spelling of "ghar"?)
- `"ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` (What is the spelling of "paani"?)
- `"ਸੂਰਜ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` (What is the spelling of "suraj"?)
- `"ਸਕੂਲ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` (What is the spelling of "school"?)

**Letter Position Questions:**
- `"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ?` (What is the first letter of "ghar"?)
- `"ਪਾਣੀ" ਦਾ ਦੂਜਾ ਅੱਖਰ ਕੀ ਹੈ?` (What is the second letter of "paani"?)

**Letter Count Questions:**
- `"ਘਰ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?` (How many letters in "ghar"?)
- `"ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?` (How many letters in "paani"?)
- `"ਕਿਤਾਬ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?` (How many letters in "kitaab"?)

**Letter Listing Questions:**
- `"ਘਰ" ਦੇ ਅੱਖਰ ਕੀ ਹਨ?` (What are the letters in "ghar"?)
- `"ਪਾਣੀ" ਨੂੰ ਅੱਖਰਾਂ ਵਿੱਚ ਵੰਡੋ?` (Break "paani" into letters?)

**Answer Format:** 
- Spelling/Letter Listing: Comma-separated characters: `ਘ, ਰ` or `ਪ, ਾ, ਣ, ੀ`
- Letter Position: Single character: `ਘ`, `ਾ`
- Letter Count: Numeric value: `2`, `4`, `5`

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

**English:**
```
What is the spelling of "behavior"? b, e, h, a, v, i, o, r.
What is the first letter in "apple"? a.
How many letters are in "cat"? 3.
Tell me which word starts with /ch/: "dog" or "chair"? chair.
```

**Hindi (हिंदी):**
```
"कमल" की वर्तनी क्या है? क, म, ल।
"घर" की वर्तनी क्या है? घ, र।
"पानी" की वर्तनी क्या है? प, आ, न, ी।
"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
"कमल" का पहला अक्षर क्या है? क।
"पानी" में कितने अक्षर हैं? 4।
```

**Bengali (বাংলা):**
```
"বই" এর বানান কী? ব, ই।
"ঘর" এর বানান কী? ঘ, র।
"জল" এর বানান কী? জ, ল।
"বই" এর প্রথম অক্ষর কী? ব।
"ঘর" এ কতটি অক্ষর আছে? 2।
```

**Tamil (தமிழ்):**
```
"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு।
"நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்।
"மலர்" எப்படி எழுதுவது? ம, ல, ர, ்।
"வீடு" இன் முதல் எழுத்து என்ன? வ।
"நீர்" இல் எத்தனை எழுத்துக்கள்? 4।
```

**Telugu (తెలుగు):**
```
"ఇల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు।
"నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు।
"పువ్వు" స్పెల్లింగ్ ఏమిటి? ప, ు, వ, ్, వ, ు।
"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి? ఇ।
"నీరు" లో ఎన్ని అక్షరాలు? 4।
```

**Gujarati (ગુજરાતી):**
```
"ઘર" ની જોડણી શું છે? ઘ, ર।
"પાણી" ની જોડણી શું છે? પ, ા, ણ, ી।
"શાળા" ની જોડણી શું છે? શ, ા, ળ, ા।
"ઘર" નો પ્રથમ અક્ષર શું છે? ઘ।
"પાણી" માં કેટલા અક્ષર છે? 4।
```

**Marathi (मराठी):**
```
"घर" ची शुद्धलेखन काय आहे? घ, र।
"पाणी" ची शुद्धलेखन काय आहे? प, ा, ण, ी।
"फूल" ची शुद्धलेखन काय आहे? फ, ू, ल।
"घर" चे पहिले अक्षर काय आहे? घ।
"पाणी" मध्ये किती अक्षरे आहेत? 4।
```

**Assamese (অসমীয়া):**
```
"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প।
"ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ।
"ঘৰ" ৰ প্ৰথম আখৰ কি? ঘ।
"পানী" ত কিমান আখৰ আছে? 4।
```

**Kannada (ಕನ್ನಡ):**
```
"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ।
"ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।
"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಮ।
"ನೀರು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4।
```

**Malayalam (മലയാളം):**
```
"വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്।
"നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।
"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? വ।
"നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।
```

**Odia (ଓଡ଼ିଆ):**
```
"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି।
"ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର।
"ବହି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ବ।
"ପାଣି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 4।
```

**Punjabi (ਪੰਜਾਬੀ):**
```
"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ।
"ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।
"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਘ।
"ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4।
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

5. **Language-Specific Questions (All Indic Languages):**
   - Quote the target word in the appropriate script
   - Answer: Unquoted comma-separated characters
   
   **Hindi Examples:**
   - `"कमल" की वर्तनी क्या है? क, म, ल।`
   - `"घर" की वर्तनी क्या है? घ, र।`
   - `"पानी" की वर्तनी क्या है? प, आ, न, ी।`
   - `"सूरज" की वर्तनी क्या है? स, ू, र, ज।`
   - `"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।`
   - `"कमल" का पहला अक्षर क्या है? क।`
   - `"पानी" में कितने अक्षर हैं? 4।`
   
   **Bengali Examples:**
   - `"বই" এর বানান কী? ব, ই।`
   - `"ঘর" এর বানান কী? ঘ, র।`
   - `"বই" এর প্রথম অক্ষর কী? ব।`
   
   **Tamil Examples:**
   - `"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு।`
   - `"நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்।`
   - `"வீடு" இன் முதல் எழுத்து என்ன? வ।`
   
   **Telugu Examples:**
   - `"ఇల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు।`
   - `"నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు।`
   - `"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి? ఇ।`
   
   **Gujarati Examples:**
   - `"ઘર" ની જોડણી શું છે? ઘ, ર।`
   - `"પાણી" ની જોડણી શું છે? પ, ા, ણ, ી।`
   - `"ઘર" નો પ્રથમ અક્ષર શું છે? ઘ।`
   
   **Marathi Examples:**
   - `"घर" ची शुद्धलेखन काय आहे? घ, र।`
   - `"पाणी" ची शुद्धलेखन काय आहे? प, ा, ण, ी।`
   - `"घर" चे पहिले अक्षर काय आहे? घ।`
   
   **Assamese Examples:**
   - `"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প।`
   - `"ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ।`
   - `"ঘৰ" ৰ প্ৰথম আখৰ কি? ঘ।`

   **Kannada Examples:**
   - `"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ।`
   - `"ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।`
   - `"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಮ।`

   **Malayalam Examples:**
   - `"വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്।`
   - `"നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।`
   - `"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? വ।`

   **Odia Examples:**
   - `"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି।`
   - `"ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର।`
   - `"ବହି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ବ।`

   **Punjabi Examples:**
   - `"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ।`
   - `"ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।`
   - `"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਘ।`

---

## Answer Format Guidelines

### Important: Language-Specific Punctuation

**All answers must end with language-specific punctuation:**
- **English:** Period (`.`) - Example: `c, a, t.`
- **All Indic Languages (Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Assamese, Kannada, Malayalam, Odia, Punjabi):** Devanagari danda (`।`) - Example: `क, म, ल।` or `ব, ই।` or `வ, ீ, ட, ு।`

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

**Hindi (हिंदी - Devanagari):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `क, म, ल।`
- Preserve script-specific formatting
- Handle combining characters correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"कमल" की वर्तनी क्या है? क, म, ल।` (Spelling - ends with `।`)
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

**Bengali (বাংলা - Bengali Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ব, ই।`
- Preserve Bengali script characters and diacritics
- Handle combining vowel marks correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"বই" এর বানান কী? ব, ই।` (Spelling - ends with `।`)
  - `"ঘর" এর বানান কী? ঘ, র।`
  - `"জল" এর বানান কী? জ, ল।`
  - `"সূর্য" এর বানান কী? স, ূ, র, ্, য।`
  - `"বিদ্যালয়" এর বানান কী? ব, ি, দ, ্, য, া, ল, য, ়।`
  - `"বই" এর প্রথম অক্ষর কী? ব।` (Letter position - ends with `।`)
  - `"জল" এ কতটি অক্ষর আছে? 2।` (Letter count - ends with `।`)

**Tamil (தமிழ் - Tamil Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `வ, ீ, ட, ு।`
- Preserve Tamil script characters and vowel signs
- Handle pulli (்) and other diacritics correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு।` (Spelling - ends with `।`)
  - `"நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்।`
  - `"மலர்" எப்படி எழுதுவது? ம, ல, ர, ்।`
  - `"பள்ளி" எப்படி எழுதுவது? ப, ள, ், ள, ி।`
  - `"சூரியன்" எப்படி எழுதுவது? ச, ூ, ர, ி, ய, ன, ்।`
  - `"வீடு" இன் முதல் எழுத்து என்ன? வ।` (Letter position - ends with `।`)
  - `"நீர்" இல் எத்தனை எழுத்துக்கள்? 4।` (Letter count - ends with `।`)

**Telugu (తెలుగు - Telugu Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ఇ, ల, ్, ల, ు।`
- Preserve Telugu script characters and vowel marks
- Handle halant (్) and other diacritics correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"ఇల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు।` (Spelling - ends with `।`)
  - `"నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు।`
  - `"పువ్వు" స్పెల్లింగ్ ఏమిటి? ప, ు, వ, ్, వ, ు।`
  - `"సూర్యుడు" స్పెల్లింగ్ ఏమిటి? స, ూ, ర, ్, య, ు, డ, ు।`
  - `"పాఠశాల" స్పెల్లింగ్ ఏమిటి? ప, ా, ఠ, శ, ా, ల।`
  - `"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి? ఇ।` (Letter position - ends with `।`)
  - `"నీరు" లో ఎన్ని అక్షరాలు? 4।` (Letter count - ends with `।`)

**Gujarati (ગુજરાતી - Gujarati Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ઘ, ર।`
- Preserve Gujarati script characters and vowel signs
- Handle combining marks correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"ઘર" ની જોડણી શું છે? ઘ, ર।` (Spelling - ends with `।`)
  - `"પાણી" ની જોડણી શું છે? પ, ા, ણ, ી।`
  - `"શાળા" ની જોડણી શું છે? શ, ા, ળ, ા।`
  - `"સૂરજ" ની જોડણી શું છે? સ, ૂ, ર, જ।`
  - `"પુસ્તક" ની જોડણી શું છે? પ, ુ, સ, ્, ત, ક।`
  - `"ઘર" નો પ્રથમ અક્ષર શું છે? ઘ।` (Letter position - ends with `।`)
  - `"પાણી" માં કેટલા અક્ષર છે? 4।` (Letter count - ends with `।`)

**Marathi (मराठी - Devanagari):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `घ, र।`
- Uses same Devanagari script as Hindi
- Preserve script-specific formatting and combining characters
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"घर" ची शुद्धलेखन काय आहे? घ, र।` (Spelling - ends with `।`)
  - `"पाणी" ची शुद्धलेखन काय आहे? प, া, ण, ी।`
  - `"फूल" ची शुद्धलेखन काय आहे? फ, ू, ल।`
  - `"सूर्य" ची शुद्धलेखन काय आहे? स, ू, র, ্, য।`
  - `"पुस्तक" ची शुद्धलेखन काय आहे? প, ু, স, ্, ত, ক।`
  - `"घर" चे पहिले अक्षर काय आहे? ঘ।` (Letter position - ends with `।`)
  - `"पाणी" मध्ये किती अक्षरे आहेत? 4।` (Letter count - ends with `।`)

**Assamese (অসমীয়া - Assamese Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ক, ি, ত, া, প।`
- Uses Bengali/Assamese script with unique characters
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প।` (Spelling - ends with `।`)
  - `"ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ।`
  - `"পানী" শব্দেৰ বানান কি? প, া, ন, ী।`
  - `"সূৰ্য" শব্দেৰ বানান কি? স, ূ, ৰ, ্, য।`
  - `"ঘৰ" ৰ প্ৰথম আখৰ কি? ঘ।` (Letter position - ends with `।`)
  - `"পানী" ত কিমান আখৰ আছে? 4।` (Letter count - ends with `।`)

**Kannada (ಕನ್ನಡ - Kannada Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ಮ, ನ, ೆ।`
- Preserve Kannada script characters and vowel signs
- Handle halant and other diacritics correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ।` (Spelling - ends with `।`)
  - `"ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।`
  - `"ಪುಸ್ತಕ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಪ, ು, ಸ, ್, ತ, ಕ।`
  - `"ಸೂರ್ಯ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಸ, ೂ, ರ, ್, ಯ।`
  - `"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಮ।` (Letter position - ends with `।`)
  - `"ನೀರು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4।` (Letter count - ends with `।`)

**Malayalam (മലയാളം - Malayalam Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `വ, ീ, ട, ്।`
- Preserve Malayalam script characters and vowel signs
- Handle chillu and other diacritics correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്।` (Spelling - ends with `।`)
  - `"നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।`
  - `"പുസ്തകം" എങ്ങനെ എഴുതുന്നു? പ, ു, സ, ്, ത, ക, ം।`
  - `"സൂര്യന്" എങ്ങനെ എഴുതുന്നു? സ, ൂ, ര, ്, യ, ന, ്।`
  - `"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? വ।` (Letter position - ends with `।`)
  - `"നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।` (Letter count - ends with `।`)

**Odia (ଓଡ଼ିଆ - Odia Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ବ, ହ, ି।`
- Preserve Odia script characters and vowel marks
- Handle consonant conjuncts correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି।` (Spelling - ends with `।`)
  - `"ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର।`
  - `"ପାଣି" ଶବ୍ଦର ବାନାନ କଣ? ପ, ା, ଣ, ି।`
  - `"ସୂର୍ଯ୍ୟ" ଶବ୍ଦର ବାନାନ କଣ? ସ, ୂ, ର, ୍, ଯ, ୍, ୟ।`
  - `"ବହି" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ବ।` (Letter position - ends with `।`)
  - `"ପାଣି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 4।` (Letter count - ends with `।`)

**Punjabi (ਪੰਜਾਬੀ - Gurmukhi Script):**
- Format: Comma-separated characters ending with Devanagari danda (`।`): `ਘ, ਰ।`
- Uses Gurmukhi script
- Handle tippi (ਂ), bindi (ਁ) and other diacritics correctly
- **Important:** Answers end with `।` (Devanagari danda), not `.` (period)
- Examples:
  - `"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ।` (Spelling - ends with `।`)
  - `"ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।`
  - `"ਕਿਤਾਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਕ, ਿ, ਤ, ਾ, ਬ।`
  - `"ਸੂਰਜ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਸ, ੂ, ਰ, ਜ।`
  - `"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਘ।` (Letter position - ends with `।`)
  - `"ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4।` (Letter count - ends with `।`)

---

## Language-Specific Considerations

### Unicode Handling

**Critical Requirements:**
- Preserve all Unicode characters correctly
- Don't break multi-byte characters
- Handle combining characters (diacritics) properly
- Maintain script-specific formatting

**Scripts to Handle:**
- **Devanagari (Hindi, Marathi):** `"कमल"`, `"घर"`, `"पानी"`, `"सूरज"`, `"विद्यालय"`, `"फूल"`, `"किताब"`, `"बच्चा"`, `"स्कूल"`, `"गाड़ी"`
- **Bengali/Assamese Script:** `"বই"`, `"ঘর"`, `"জল"`, `"সূর্য"`, `"বিদ্যালয়"`, `"ফুল"`, `"গাছ"`, `"শিশু"`, `"স্কুল"`, `"মানুষ"`, `"কিতাপ"`, `"পানী"`
- **Tamil Script:** `"புத்தகம்"`, `"வீடு"`, `"நீர்"`, `"சூரியன்"`, `"பள்ளி"`, `"மலர்"`, `"மரம்"`, `"குழந்தை"`, `"பாடசாலை"`
- **Telugu Script:** `"పుస్తకం"`, `"ఇల్లు"`, `"నీరు"`, `"సూర్యుడు"`, `"పాఠశాల"`, `"పువ్వు"`, `"చెట్టు"`, `"పిల్ల"`, `"మనిషి"`
- **Kannada Script:** `"ಪುಸ್ತಕ"`, `"ಮನೆ"`, `"ನೀರು"`, `"ಸೂರ್ಯ"`, `"ಶಾಲೆ"`, `"ಹೂವು"`, `"ಮರ"`, `"ಮಗು"`, `"ಮನುಷ್ಯ"`
- **Malayalam Script:** `"പുസ്തകം"`, `"വീട്"`, `"നീര്"`, `"സൂര്യന്"`, `"പാഠശാല"`, `"പൂവ്"`, `"മരം"`, `"കുട്ടി"`, `"മനുഷ്യന്"`
- **Gujarati Script:** `"પુસ્તક"`, `"ઘર"`, `"પાણી"`, `"સૂરજ"`, `"શાળા"`, `"ફૂલ"`, `"ઝાડ"`, `"બાળક"`, `"માણસ"`
- **Odia Script:** `"ବହି"`, `"ଘର"`, `"ପାଣି"`, `"ସୂର୍ଯ୍ୟ"`, `"ବିଦ୍ୟାଳୟ"`, `"ଫୁଲ"`, `"ଗଛ"`, `"ପିଲା"`, `"ମଣିଷ"`
- **Gurmukhi (Punjabi):** `"ਕਿਤਾਬ"`, `"ਘਰ"`, `"ਪਾਣੀ"`, `"ਸੂਰਜ"`, `"ਸਕੂਲ"`, `"ਫੁੱਲ"`, `"ਰੁੱਖ"`, `"ਬੱਚਾ"`, `"ਇਨਸਾਨ"`
- **Arabic:** Handle right-to-left text correctly
- **Chinese/Japanese:** Handle multi-character words
- **Any script:** Preserve script-specific formatting

### Script-Specific Patterns

**Devanagari (Hindi, Marathi):**
- Question patterns:
  - **Hindi:** `"[word]" की वर्तनी क्या है?` (spelling), `"[word]" का [position] अक्षर क्या है?` (position), `"[word]" में कितने अक्षर हैं?` (count), `"[word]" के अक्षर क्या हैं?` (listing)
  - **Marathi:** `"[word]" ची शुद्धलेखन काय आहे?` (spelling), `"[word]" चे [position] अक्षर काय आहे?` (position), `"[word]" मध्ये किती अक्षरे आहेत?` (count), `"[word]" ची अक्षरे काय आहेत?` (listing)
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `क, म, ल`
  - Letter position: Single character: `क`
  - Letter count: Numeric: `3`
- Examples:
- Examples:
  - `"कमल" की वर्तनी क्या है? क, म, ल।`
  - `"घर" की वर्तनी क्या है? घ, र।`
  - `"पानी" की वर्तनी क्या है? प, आ, न, ी।`
  - `"सूरज" की वर्तनी क्या है? स, ू, र, ज।`
  - `"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।`
  - `"फूल" की वर्तनी क्या है? फ, ू, ल।`
  - `"किताब" की वर्तनी क्या है? क, ि, त, ा, ब।`
  - `"बच्चा" की वर्तनी क्या है? ब, च, ्, च, ा।`
  - `"स्कूल" की वर्तनी क्या है? स, ्, क, ू, ल।`
  - `"गाड़ी" की वर्तनी क्या है? ग, ा, ड, ्, र, ी।`
  - `"सपना" की वर्तनी क्या है? स, प, न, ा।`
  - `"दोस्त" की वर्तनी क्या है? द, ो, स, ्, त।`
  - `"खेल" की वर्तनी क्या है? ख, े, ल।`
  - `"प्यार" की वर्तनी क्या है? प, ्, य, ा, र।`
  - `"नदी" की वर्तनी क्या है? न, द, ी।`
  - `"पहाड़" की वर्तनी क्या है? प, ह, ा, ड, ्।`
  - `"आकाश" की वर्तनी क्या है? आ, क, ा, श।`
  - `"धरती" की वर्तनी क्या है? ध, र, त, ी।`
  - `"हवा" की वर्तनी क्या है? ह, व, ा।`
  - `"आग" की वर्तनी क्या है? आ, ग।`
  - `"कमल" का पहला अक्षर क्या है? क।`
  - `"कमल" का दूसरा अक्षर क्या है? म।`
  - `"घर" का पहला अक्षर क्या है? घ।`
  - `"पानी" का तीसरा अक्षर क्या है? न।`
  - `"सूरज" का अंतिम अक्षर क्या है? ज।`
  - `"किताब" का दूसरा अक्षर क्या है? ि।`
  - `"पानी" में कितने अक्षर हैं? 4।`
  - `"घर" में कितने अक्षर हैं? 2।`
  - `"विद्यालय" में कितने अक्षर हैं? 8।`
  - `"बच्चा" में कितने अक्षर हैं? 5।`
  - `"कमल" के अक्षर क्या हैं? क, म, ल।`
  - `"घर" के अक्षर क्या हैं? घ, र।`

  - `"घर" ची शुद्धलेखन काय आहे? घ, र।` (Marathi)
  - `"पाणी" ची शुद्धलेखन काय आहे? प, ा, ण, ी।`
  - `"फूल" ची शुद्धलेखन काय आहे? फ, ू, ल।`
  - `"पुस्तक" ची शुद्धलेखन काय आहे? प, ु, स, ्, त, क।`
  - `"शाळा" ची शुद्धलेखन काय आहे? श, ा, ळ, ा।`
  - `"झाड" ची शुद्धलेखन काय आहे? झ, ा, ड।`
  - `"सूर्य" ची शुद्धलेखन काय आहे? स, ू, र, ्, य।`
  - `"अन्न" ची शुद्धलेखन काय आहे? अ, न, ्, न।`
  - `"मित्र" ची शुद्धलेखन काय आहे? म, ि, त, ्, र।`
  - `"प्रेम" ची शुद्धलेखन काय आहे? प, ्, र, े, म।`
  - `"घर" चे पहिले अक्षर काय आहे? घ।`
  - `"पाणी" चे दुसरे अक्षर काय आहे? ा।`
  - `"शाळा" चे शेवटचे अक्षर काय आहे? ा।`
  - `"पुस्तक" चे पहिले अक्षर काय आहे? प।`
  - `"घर" मध्ये किती अक्षरे आहेत? 2।`
  - `"पाणी" मध्ये किती अक्षरे आहेत? 4।`
  - `"शाळा" मध्ये किती अक्षरे आहेत? 4।`
  - `"पुस्तक" मध्ये किती अक्षरे आहेत? 6।`
  - `"घर" ची अक्षरे काय आहेत? घ, र।`
  - `"पाणी" ची अक्षरे काय आहेत? प, ा, ण, ी।`

**Bengali (বাংলা):**
- Question patterns:
  - Spelling: `"[word]" এর বানান কী?`
  - Letter position: `"[word]" এর [position] অক্ষর কী?`
  - Letter count: `"[word]" এ কতটি অক্ষর আছে?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ব, ই`
  - Letter position: Single character: `ব`
  - Letter count: Numeric: `2`
- Examples:
  - `"বই" এর বানান কী? ব, ই।`
  - `"ঘর" এর বানান কী? ঘ, র।`
  - `"জল" এর বানান কী? জ, ল।`
  - `"সূর্য" এর বানান কী? স, ূ, র, ্, য।`
  - `"বিদ্যালয়" এর বানান কী? ব, ি, দ, ্, য, া, ল, য়।`
  - `"ফুল" এর বানান কী? ফ, ু, ল।`
  - `"গাছ" এর বানান কী? গ, া, ছ।`
  - `"পাখি" এর বানান কী? প, া, খ, ি।`
  - `"আকাশ" এর বানান কী? আ, ক, া, শ।`
  - `"নদী" এর বানান কী? ন, দ, ী।`
  - `"মাছ" এর বানান কী? ম, া, ছ।`
  - `"ভাত" এর বানান কী? ভ, া, ত।`
  - `"কলম" এর বানান কী? ক, ল, ম।`
  - `"বই" এর প্রথম অক্ষর কী? ব।`
  - `"ঘর" এর শেষ অক্ষর কী? র।`
  - `"জল" এর প্রথম অক্ষর কী? জ।`
  - `"সূর্য" এর দ্বিতীয় অক্ষর কী? ূ।`
  - `"আকাশ" এর তৃতীয় অক্ষর কী? া।`
  - `"বই" এ কতটি অক্ষর আছে? 2।`
  - `"ঘর" এ কতটি অক্ষর আছে? 2।`
  - `"বিদ্যালয়" এ কতটি অক্ষর আছে? 8।`
  - `"ফুল" এ কতটি অক্ষর আছে? 3।`
  - `"গাছ" এ কতটি অক্ষর আছে? 3।`

  - `"ভয়" এর বানান কী? ভ, য়।` (Bengali)
  - `"সহজ" এর বানান কী? স, হ, জ।` (Bengali)
  - `"সময়" এর বানান কী? স, ম, য়।` (Bengali)

  - `"ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ।` (Assamese)
  - `"পানী" শব্দেৰ বানান কি? প, া, ন, ী।`
  - `"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প।`
  - `"সূৰ্য" শব্দেৰ বানান কি? স, ূ, ৰ, ্, য।`
  - `"বিদ্যালয়" শব্দেৰ বানান কি? ব, ি, দ, ্, য, া, ল, য়।`
  - `"ফুল" শব্দেৰ বানান কি? ফ, ু, ল।`
  - `"গছ" শব্দেৰ বানান কি? গ, ছ।`
  - `"চৰাই" শব্দেৰ বানান কি? চ, ৰ, া, ই।`
  - `"আকাশ" শব্দেৰ বানান কি? আ, ক, া, শ।`
  - `"নদী" শব্দেৰ বানান কি? ন, দ, ী।`
  - `"মাছ" শব্দেৰ বানান কি? ম, া, ছ।`
  - `"ভাত" শব্দেৰ বানান কি? ভ, া, ত।`
  - `"কলম" শব্দেৰ বানান কি? ক, ল, ম।`
  - `"ঘৰ" ৰ প্ৰথম আখৰ কি? ঘ।`
  - `"পানী" ৰ দ্বিতীয় আখৰ কি? া।`
  - `"কিতাপ" ৰ শেষ আখৰ কি? প।`
  - `"সূৰ্য" ৰ প্ৰথম আখৰ কি? স।`
  - `"আকাশ" ৰ তৃতীয় আখৰ কি? া।`
  - `"ঘৰ" ত কিমান আখৰ আছে? 2।`
  - `"পানী" ত কিমান আখৰ আছে? 4।`
  - `"কিতাপ" ত কিমান আখৰ আছে? 5।`
  - `"ফুল" ত কিমান আখৰ আছে? 3।`
  - `"গছ" ত কিমান আখৰ আছে? 2।`

**Tamil (தமிழ்):**
- Question patterns:
  - Spelling: `"[word]" எப்படி எழுதுவது?`
  - Letter position: `"[word]" இன் [position] எழுத்து என்ன?`
  - Letter count: `"[word]" இல் எத்தனை எழுத்துக்கள்?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `வ, ீ, ட, ு`
  - Letter position: Single character: `வ`
  - Letter count: Numeric: `4`
- Examples:
  - `"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு।`
  - `"நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்।`
  - `"மலர்" எப்படி எழுதுவது? ம, ல, ர, ்।`

**Telugu (తెలుగు):**
- Question patterns:
  - Spelling: `"[word]" స్పెల్లింగ్ ఏమిటి?`
  - Letter position: `"[word]" యొక్క [position] అక్షరం ఏమిటి?`
  - Letter count: `"[word]" లో ఎన్ని అక్షరాలు?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ఇ, ల, ్, ల, ు`
  - Letter position: Single character: `ఇ`
  - Letter count: Numeric: `5`
- Examples:
  - `"வீடு" எப்படி எழுதுவது? வ, ீ, ட, ு।`
  - `"நீர்" எப்படி எழுதுவது? ந, ீ, ர, ்।`
  - `"மலர்" எப்படி எழுதுவது? ம, ல, ர, ்।`
  - `"சூரியன்" எப்படி எழுதுவது? ச, ூ, ர, ி, ய, ன, ்।`
  - `"பள்ளி" எப்படி எழுதுவது? ப, ள, ், ள, ி।`
  - `"மரம்" எப்படி எழுதுவது? ம, ர, ம, ்।`
  - `"புத்தகம்" எப்படி எழுதுவது? ப, ு, த, ், த, க, ம, ்।`
  - `"குழந்தை" எப்படி எழுதுவது? க, ு, ழ, ந, ், த, ை।`
  - `"பூ" எப்படி எழுதுவது? ப, ூ।`
  - `"பால்" எப்படி எழுதுவது? ப, ா, ல, ்।`
  - `"அம்மா" எப்படி எழுதுவது? அ, ம, ், ம, ா।`
  - `"அப்பா" எப்படி எழுதுவது? அ, ப, ், ப, ா।`
  - `"கடல்" எப்படி எழுதுவது? க, ட, ல, ்।`
  - `"வானம்" எப்படி எழுதுவது? வ, ா, ன, ம, ்।`
  - `"நிலா" எப்படி எழுதுவது? ந, ி, ல, ா।`
  - `"வீடு" இன் முதல் எழுத்து என்ன? வ।`
  - `"நீர்" இன் இரண்டாம் எழுத்து என்ன? ீ।`
  - `"மலர்" இன் கடைசி எழுத்து என்ன? ்।`
  - `"சூரியன்" இன் முதல் எழுத்து என்ன? ச।`
  - `"மரம்" இன் மூன்றாம் எழுத்து என்ன? ம।`
  - `"வீடு" இல் எத்தனை எழுத்துக்கள்? 4।`
  - `"நீர்" இல் எத்தனை எழுத்துக்கள்? 4।`
  - `"மலர்" இல் எத்தனை எழுத்துக்கள்? 4।`
  - `"பூ" இல் எத்தனை எழுத்துக்கள்? 2।`
  - `"புத்தகம்" இல் எத்தனை எழுத்துக்கள்? 8।`

  - `"இల్లు" స్పెల్లింగ్ ఏమిటి? ఇ, ల, ్, ల, ు।`
  - `"నీరు" స్పెల్లింగ్ ఏమిటి? న, ీ, ర, ు।`
  - `"పువ్వు" స్పెల్లింగ్ ఏమిటి? ప, ు, వ, ్, వ, ు।`
  - `"సూర్యుడు" స్పెల్లింగ్ ఏమిటి? స, ూ, ర, ్, య, ు, డ, ు।`
  - `"పాఠశాల" స్పెల్లింగ్ ఏమిటి? ప, ా, ఠ, శ, ా, ల।`
  - `"చెట్టు" స్పెల్లింగ్ ఏమిటి? చ, ె, ట, ్, ట, ు।`
  - `"పుస్తకం" స్పెల్లింగ్ ఏమిటి? ప, ు, స, ్, త, క, ం।`
  - `"పిల్ల" స్పెల్లింగ్ ఏమిటి? ప, ి, ల, ్, ల।`
  - `"అమ్మ" స్పెల్లింగ్ ఏమిటి? అ, మ, ్, మ।`
  - `"నాన్న" స్పెల్లింగ్ ఏమిటి? న, ా, న, ్, న।`
  - `"ఆకాశం" స్పెల్లింగ్ ఏమిటి? ఆ, క, ా, శ, ం।`
  - `"సముద్రం" స్పెల్లింగ్ ఏమిటి? స, మ, ు, ద, ్, ర, ం।`
  - `"నేల" స్పెల్లింగ్ ఏమిటి? న, ే, ల।`
  - `"గాలి" స్పెల్లింగ్ ఏమిటి? గ, ా, ల, ి।`
  - `"నిప్పు" స్పెల్లింగ్ ఏమిటి? న, ి, ప, ్, ప, ు।`
  - `"ఇల్లు" యొక్క మొదటి అక్షరం ఏమిటి? ఇ।`
  - `"నీరు" యొక్క రెండవ అక్షరం ఏమిటి? ీ।`
  - `"పువ్వు" యొక్క చివరి అక్షరం ఏమిటి? ు।`
  - `"చెట్టు" యొక్క మొదటి అక్షరం ఏమిటి? చ।`
  - `"పాఠశాల" యొక్క మూడవ అక్షరం ఏమిటి? ఠ।`
  - `"ఇల్లు" లో ఎన్ని అక్షరాలు? 5।`
  - `"నీరు" లో ఎన్ని అక్షరాలు? 4।`
  - `"పువ్వు" లో ఎన్ని అక్షరాలు? 6।`
  - `"సూర్యుడు" లో ఎన్ని అక్షరాలు? 8।`
  - `"చెట్టు" లో ఎన్ని అక్షరాలు? 6।`

**Gujarati (ગુજરાતી):**
- Question patterns:
  - Spelling: `"[word]" ની જોડણી શું છે?`
  - Letter position: `"[word]" નો [position] અક્ષર શું છે?`
  - Letter count: `"[word]" માં કેટલા અક્ષર છે?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ઘ, ર`
  - Letter position: Single character: `ઘ`
  - Letter count: Numeric: `2`
- Examples:
  - `"ઘર" ની જોડણી શું છે? ઘ, ર।`
  - `"પાણી" ની જોડણી શું છે? પ, ા, ણ, ી।`
  - `"સૂરજ" ની જોડણી શું છે? સ, ૂ, ર, જ।`
  - `"શાળા" ની જોડણી શું છે? શ, ા, ળ, ા।`
  - `"પુસ્તક" ની જોડણી શું છે? પ, ુ, સ, ્, ત, ક।`
  - `"ફૂલ" ની જોડણી શું છે? ફ, ૂ, લ।`
  - `"ઝાડ" ની જોડણી શું છે? ઝ, ા, ડ।`
  - `"બાળક" ની જોડણી શું છે? બ, ા, ળ, ક।`
  - `"કામ" ની જોડણી શું છે? ક, ા, મ।`
  - `"નામ" ની જોડણી શું છે? ન, ા, મ।`
  - `"ગામ" ની જોડણી શું છે? ગ, ા, મ।`
  - `"રમત" ની જોડણી શું છે? ર, મ, ત।`
  - `"આકાશ" ની જોડણી શું છે? આ, ક, ા, શ।`
  - `"પવન" ની જોડણી શું છે? પ, વ, ન।`
  - `"દરિયો" ની જોડણી શું છે? દ, ર, િ, ય, ો।`
  - `"ઘર" નો પ્રથમ અક્ષર શું છે? ઘ।`
  - `"પાણી" નો બીજો અક્ષર શું છે? ા।`
  - `"શાળા" નો છેલ્લો અક્ષર શું છે? ા।`
  - `"સૂરજ" નો પ્રથમ અક્ષર શું છે? સ।`
  - `"પુસ્તક" નો ત્રીજો અક્ષર શું છે? સ।`
  - `"ઘર" માં કેટલા અક્ષર છે? 2।`
  - `"પાણી" માં કેટલા અક્ષર છે? 4।`
  - `"શાળા" માં કેટલા અક્ષર છે? 4।`
  - `"ફૂલ" માં કેટલા અક્ષર છે? 3।`
  - `"પુસ્તક" માં કેટલા અક્ષર છે? 6।`

**Assamese (অসমীয়া):**
- Question patterns:
  - Spelling: `"[word]" শব্দেৰ বানান কি?`
  - Letter position: `"[word]" ৰ [position] আখৰ কি?`
  - Letter count: `"[word]" ত কিমান আখৰ আছে?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ঘ, ৰ`
  - Letter position: Single character: `ঘ`
  - Letter count: Numeric: `2`
- Examples:
  - `"ঘৰ" শব্দেৰ বানান কি? ঘ, ৰ।`
  - `"পানী" শব্দেৰ বানান কি? প, া, ন, ী।`
  - `"কিতাপ" শব্দেৰ বানান কি? ক, ি, ত, া, প।`
  - `"সূৰ্য" শব্দেৰ বানান কি? স, ূ, ৰ, ্, য।`
  - `"বিদ্যালয়" শব্দেৰ বানান কি? ব, ি, দ, ্, য, া, ল, য়।`
  - `"ফুল" শব্দেৰ বানান কি? ফ, ু, ল।`
  - `"গছ" শব্দেৰ বানান কি? গ, ছ।`
  - `"চৰাই" শব্দেৰ বানান কি? চ, ৰ, া, ই।`
  - `"আকাশ" শব্দেৰ বানান কি? আ, ক, া, শ।`
  - `"নদী" শব্দেৰ বানান কি? ন, দ, ী।`
  - `"মাছ" শব্দেৰ বানান কি? ম, া, ছ।`
  - `"ভাত" শব্দেৰ বানান কি? ভ, া, ত।`
  - `"কলম" শব্দেৰ বানান কি? ক, ল, ম।`
  - `"ঘৰ" ৰ প্ৰথম আখৰ কি? ঘ।`
  - `"পানী" ৰ দ্বিতীয় আখৰ কি? া।`
  - `"কিতাপ" ৰ শেষ আখৰ কি? প।`
  - `"সূৰ্য" ৰ প্ৰথম আখৰ কি? স।`
  - `"আকাশ" ৰ তৃতীয় আখৰ কি? া।`
  - `"ঘৰ" ত কিমান আখৰ আছে? 2।`
  - `"পানী" ত কিমান আখৰ আছে? 4।`
  - `"কিতাপ" ত কিমান আখৰ আছে? 5।`
  - `"ফুল" ত কিমান আখৰ আছে? 3।`
  - `"গছ" ত কিমান আখৰ আছে? 2।`

**Kannada (ಕನ್ನಡ):**
- Question patterns:
  - Spelling: `"[word]" ಸ್ಪೆಲಿಂಗ್ ಏನು?`
  - Letter position: `"[word]" ಯ [position] ಅಕ್ಷರ ಏನು?`
  - Letter count: `"[word]" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ಮ, ನ, ೆ`
  - Letter position: Single character: `ಮ`
  - Letter count: Numeric: `3`
- Examples:
  - `"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ನ, ೆ।`
  - `"ನೀರು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ೀ, ರ, ು।`
  - `"ಶಾಲೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಶ, ಾ, ಲ, ೆ।`
  - `"ಪುಸ್ತಕ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಪ, ು, ಸ, ್, ತ, ಕ।`
  - `"ಹೂವು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಹ, ೂ, ವ, ು।`
  - `"ಮರ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ರ।`
  - `"ಸೂರ್ಯ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಸ, ೂ, ರ, ್, ಯ।`
  - `"ಮಗು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಮ, ಗ, ು।`
  - `"ಹಾಲು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಹ, ಾ, ಲ, ು।`
  - `"ಊಟ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಊ, ಟ।`
  - `"ಆಟ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಆ, ಟ।`
  - `"ನದಿ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ನ, ದ, ಿ।`
  - `"ಆಕಾಶ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಆ, ಕ, ಾ, ಶ।`
  - `"ಭೂಮಿ" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಭ, ೂ, ಮ, ಿ।`
  - `"ಕಾಡು" ಸ್ಪೆಲಿಂಗ್ ಏನು? ಕ, ಾ, ಡ, ು।`
  - `"ಮನೆ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಮ।`
  - `"ನೀರು" ಯ ಎರಡನೇ ಅಕ್ಷರ ಏನು? ೀ।`
  - `"ಶಾಲೆ" ಯ ಕೊನೆಯ ಅಕ್ಷರ ಏನು? ೆ।`
  - `"ಪುಸ್ತಕ" ಯ ಮೊದಲ ಅಕ್ಷರ ಏನು? ಪ।`
  - `"ಹೂವು" ಯ ಮೂರನೇ ಅಕ್ಷರ ಏನು? ವ।`
  - `"ಮನೆ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 3।`
  - `"ನೀರು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4।`
  - `"ಶಾಲೆ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4।`
  - `"ಪುಸ್ತಕ" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 6।`
  - `"ಹೂವು" ಯಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ? 4।`

**Malayalam (മലയാളം):**
- Question patterns:
  - Spelling: `"[word]" എങ്ങനെ എഴുതുന്നു?`
  - Letter position: `"[word]" ന്റെ [position] അക്ഷരം എന്താണ്?`
  - Letter count: `"[word]" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `വ, ീ, ട, ്`
  - Letter position: Single character: `വ`
  - Letter count: Numeric: `4`
- Examples:
  - `"വീട്" എങ്ങനെ എഴുതുന്നു? വ, ീ, ട, ്।`
  - `"നീര്" എങ്ങനെ എഴുതുന്നു? ന, ീ, ര, ്।`
  - `"പുസ്തകം" എങ്ങനെ എഴുതുന്നു? പ, ു, സ, ്, ത, ക, ം।`
  - `"സ്കൂൾ" എങ്ങനെ എഴുതുന്നു? സ, ്, ക, ൂ, ള, ്।`
  - `"മരം" എങ്ങനെ എഴുതുന്നു? മ, ര, ം।`
  - `"പൂവ്" എങ്ങനെ എഴുതുന്നു? പ, ൂ, വ, ്।`
  - `"സൂര്യൻ" എങ്ങനെ എഴുതുന്നു? സ, ൂ, ര, ്യ, ന, ്।`
  - `"കുട്ടി" എങ്ങനെ എഴുതുന്നു? ക, ു, ട, ്ട, ി।`
  - `"അമ്മ" എങ്ങനെ എഴുതുന്നു? അ, മ, ്, മ।`
  - `"അച്ഛൻ" എങ്ങനെ എഴുതുന്നു? അ, ച, ്,ഛ, ന, ്।`
  - `"കടൽ" എങ്ങനെ എഴുതുന്നു? ക, ട, ല, ്।`
  - `"ആകാശം" എങ്ങനെ എഴുതുന്നു? ആ, ക, ാ, ശ, ം।`
  - `"മഴ" എങ്ങനെ എഴുതുന്നു? മ, ഴ।`
  - `"പുഴ" എങ്ങനെ എഴുതുന്നു? പ, ു, ഴ।`
  - `"കാട്" എങ്ങനെ എഴുതുന്നു? ക, ാ, ട, ്।`
  - `"വീട്" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? വ।`
  - `"നീര്" ന്റെ രണ്ടാമത്തെ അക്ഷരം എന്താണ്? ീ।`
  - `"പുസ്തകം" ന്റെ അവസാനത്തെ അക്ഷരം എന്താണ്? ം।`
  - `"മരം" ന്റെ ആദ്യത്തെ അക്ഷരം എന്താണ്? മ।`
  - `"പൂവ്" ന്റെ മൂന്നാമത്തെ അക്ഷരം എന്താണ്? വ।`
  - `"വീട്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।`
  - `"നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।`
  - `"പുസ്തകം" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 7।`
  - `"മരം" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 3।`
  - `"പൂവ്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।`

**Odia (ଓଡ଼ିଆ):**
- Question patterns:
  - Spelling: `"[word]" ଶବ୍ଦର ବାନାନ କଣ?`
  - Letter position: `"[word]" ର [position] ଅକ୍ଷର କଣ?`
  - Letter count: `"[word]" ରେ କେତେ ଅକ୍ଷର ଅଛି?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ଘ, ର`
  - Letter position: Single character: `ଘ`
  - Letter count: Numeric: `2`
- Examples:
  - `"ଘର" ଶବ୍ଦର ବାନାନ କଣ? ଘ, ର।`
  - `"ପାଣି" ଶବ୍ଦର ବାନାନ କଣ? ପ, ା, ଣ, ି।`
  - `"ବହି" ଶବ୍ଦର ବାନାନ କଣ? ବ, ହ, ି।`
  - `"ବିଦ୍ୟାଳୟ" ଶବ୍ଦର ବାନାନ କଣ? ବ, ି, ଦ, ୍, ଯ, ା, ଳ, ଯ, ୍।`
  - `"ସୂର୍ଯ୍ୟ" ଶବ୍ଦର ବାନାନ କଣ? ସ, ୂ, ର, ୍, ଯ, ୍, ଯ।`
  - `"ଫୁଲ" ଶବ୍ଦର ବାନାନ କଣ? ଫ, ୁ, ଲ।`
  - `"ଗଛ" ଶବ୍ଦର ବାନାନ କଣ? ଗ, ଛ।`
  - `"ଆକାଶ" ଶବ୍ଦର ବାନାନ କଣ? ଆ, କ, ା, ଶ।`
  - `"ନଦୀ" ଶବ୍ଦର ବାନାନ କଣ? ନ, ଦ, ী।`
  - `"ମାଛ" ଶବ୍ଦର ବାନାନ କଣ? ମ, া, ଛ।`
  - `"ଭାତ" ଶବ୍ଦର ବାନାନ କଣ? ଭ, ା, ତ।`
  - `"କଲମ" ଶବ୍ଦର ବାନାନ କଣ? କ, ଲ, ମ।`
  - `"ପକ୍ଷୀ" ଶବ୍ଦର ବାନାନ କଣ? ପ, କ, ୍, ଷ, ୀ।`
  - `"ମଣିଷ" ଶବ୍ଦର ବାନାନ କଣ? ମ, ଣ, ି, ଷ।`
  - `"ପିଲା" ଶବ୍ଦର ବାନାନ କଣ? ପ, ି, ଲ, ା।`
  - `"ଘର" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ଘ।`
  - `"ପାଣି" ର ଦ୍ୱିତୀୟ ଅକ୍ଷର କଣ? ା।`
  - `"ବହି" ର ଶେଷ ଅକ୍ଷର କଣ? ି।`
  - `"ସୂର୍ଯ୍ୟ" ର ପ୍ରଥମ ଅକ୍ଷର କଣ? ସ।`
  - `"ଆକାଶ" ର ତୃତୀୟ ଅକ୍ଷର କଣ? ା।`
  - `"ଘର" ରେ କେତେ ଅକ୍ଷର ଅଛି? 2।`
  - `"ପାଣି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 4।`
  - `"ବହି" ରେ କେତେ ଅକ୍ଷର ଅଛି? 3।`
  - `"ଫୁଲ" ରେ କେତେ ଅକ୍ଷର ଅଛି? 3।`
  - `"ଗଛ" ରେ କେତେ ଅକ୍ଷର ଅଛି? 2।`

**Punjabi (ਪੰਜਾਬੀ):**
- Question patterns:
  - Spelling: `"[word]" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?`
  - Letter position: `"[word]" ਦਾ [position] ਅੱਖਰ ਕੀ ਹੈ?`
  - Letter count: `"[word]" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?`
- Answer formats:
  - Spelling/Listing: Comma-separated characters: `ਘ, ਰ`
  - Letter position: Single character: `ਘ`
  - Letter count: Numeric: `2`
- Examples:
  - `"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਘ, ਰ।`
  - `"ਪਾਣੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਾ, ਣ, ੀ।`
  - `"ਕਿਤਾਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਕ, ਿ, ਤ, ਾ, ਬ।`
  - `"ਸਕੂਲ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਸ, ਕ, ੂ, ਲ।`
  - `"ਸੂਰਜ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਸ, ੂ, ਰ, ਜ।`
  - `"ਫੁੱਲ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਫ, ੁ, ੱ, ਲ।`
  - `"ਰੁੱਖ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਰ, ੁ, ੱ, ਖ।`
  - `"ਬੱਚਾ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਬ, ੱ, ਚ, ਾ।`
  - `"ਅਸਮਾਨ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਅ, ਸ, ਮ, ਾ, ਨ।`
  - `"ਹਵਾ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਹ, ਵ, ਾ।`
  - `"ਧਰਤੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਧ, ਰ, ਤ, ੀ।`
  - `"ਜਲ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਜ, ਲ।`
  - `"ਮਾਂ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਮ, ਾ, ਂ।`
  - `"ਪਿਤਾ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਪ, ਿ, ਤ, ਾ।`
  - `"ਰੋਟੀ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਰ, ੋ, ਟ, ੀ।`
  - `"ਘਰ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਘ।`
  - `"ਪਾਣੀ" ਦਾ ਦੂਜਾ ਅੱਖਰ ਕੀ ਹੈ? ਾ।`
  - `"ਕਿਤਾਬ" ਦਾ ਤੀਜਾ ਅੱਖਰ ਕੀ ਹੈ? ਤ।`
  - `"ਸੂਰਜ" ਦਾ ਆਖਰੀ ਅੱਖਰ ਕੀ ਹੈ? ਜ।`
  - `"ਫੁੱਲ" ਦਾ ਪਹਿਲਾ ਅੱਖਰ ਕੀ ਹੈ? ਫ।`
  - `"ਘਰ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 2।`
  - `"ਪਾਣੀ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4।`
  - `"ਕਿਤਾਬ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 5।`
  - `"ਸਕੂਲ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4।`
  - `"ਰੁੱਖ" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ? 4।`

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
6. **Language-Specific Punctuation:** Use appropriate sentence terminators (e.g., `.` for English, `।` for Indic languages)

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

### 6. Complex Scripts & Character Segmentation (Indic)

**Problem:** Indic scripts (Abugidas) involve complex visual rendering where one visual unit (syllable/akshar) often consists of multiple logical Unicode characters (consonants, vowel signs, halant/virama).

**Solution:** 
- Segment words into their constituent **logical Unicode characters**. 
- Explicitly list individual consonants, vowel signs (matras), and modifiers (halant/virama, nukta, chandrabindu, tippi, etc.) as separate items.
- Do NOT group them by visual syllable.

**Example (Hindi - Conjuncts):**
```
"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य।
```
*(Breakdown: Consonant `व` + Vowel Sign `ि` + Consonant `द` + Halant `्` + Consonant `य` + Vowel Sign `ा` + Consonant `ल` + Consonant `य`)*

**Example (Malayalam - Virama):**
```
"നീര്" ല് എത്ര അക്ഷരങ്ങൾ ഉണ്ട്? 4।
```
*(Breakdown: `ന` + `ീ` + `ര` + `്`)*

**Example (Punjabi - Modifiers):**
```
"ਅੰਬ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ? ਅ, ੰ, ਬ।
```
*(Breakdown: `ਅ` + Tippi `ੰ` + `ਬ`)*

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
- [ ] Multi-character words and diacritics are segmented correctly (logical Unicode characters)
- [ ] Script-specific formatting is maintained
- [ ] **Indic Languages:** Answers must end with Devanagari danda (`।`)
- [ ] **Indic Languages:** No mixing of scripts within a single word (unless intended)

**Answer Format:**
- [ ] Spelling answers are comma-separated letters: `c, a, t`
- [ ] Letter position answers are single letters: `a`
- [ ] Letter count answers are numeric: `3`
- [ ] Sound matching answers are unquoted words: `chair`
- [ ] Indic answers follow logical character segmentation rules

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

8. **Indic Character Segmentation:**
   - Input: `What is the letter count of "विद्यालय"?`
   - Expected Check: Count should be 8 logical characters (`व, ि, द, ्, य, ा, ल, य`), NOT 4 visual syllables.

9. **Indic Sentence Termination:**
   - Input: `"घर" की वर्तनी क्या है? घ, र。` (incorrect punctuation)
   - Expected Fix: `"घर" की वर्तनी क्या है? घ, र।` (ends with `।`)

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
- [ ] **Indic:** Answers end with Danda (`।`)
- [ ] **Indic:** Words are correctly segmented

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
### Mistake 11: Incorrect Indic Punctuation or Segmentation
 
**Problem:**
```
"घर" की वर्तनी क्या है? घ, र. (Period instead of Danda)
"विद्यालय" की वर्तनी क्या है? वि, द, य, ा, ल, य। (Visual syllable segmentation)
```
 
**Fix:**
```
"घर" की वर्तनी क्या है? घ, र। (Use Danda)
"विद्यालय" की वर्तनी क्या है? व, ि, द, ्, य, ा, ल, य। (Logical Unicode segmentation)
```

---

## Summary

### Dataset Format

**Structure:** 
- **English:** `Q? A. Q? A. Q? A. ...`
- **Indic Languages (Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Assamese, Kannada, Malayalam, Odia, Punjabi):** `Q? A। Q? A। Q? A। ...`

**Key Points:**
- Question ends with `?` followed by space
- Answer ends with language-specific punctuation:
  - **English:** `.` (period)
  - **All Indic Languages:** `।` (Devanagari danda)
- Pairs separated by punctuation + space:
  - **English:** `. ` (period + space)
  - **Indic Languages:** `। ` (danda + space)
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

**Examples (Bengali):**
- `বই এর বানান কী?` → `"বই" এর বানান কী?`
- `ঘর এর বানান কী?` → `"ঘর" এর বানান কী?`

**Examples (Tamil):**
- `வீடு எப்படி எழுதுவது?` → `"வீடு" எப்படி எழுதுவது?`
- `நீர் எப்படி எழுதுவது?` → `"நீர்" எப்படி எழுதுவது?`

**Examples (Telugu):**
- `ఇల్లు స్పెల్లింగ్ ఏమిటి?` → `"ఇల్లు" స్పెల్లింగ్ ఏమిటి?`
- `నీరు స్పెల్లింగ్ ఏమిటి?` → `"నీరు" స్పెల్లింగ్ ఏమిటి?`

**Examples (Gujarati):**
- `ઘર ની જોડણી શું છે?` → `"ઘર" ની જોડણી શું છે?`
- `પાણી ની જોડણી શું છે?` → `"પાણી" ની જોડણી શું છે?`

**Examples (Marathi):**
- `घर ची शुद्धलेखन काय आहे?` → `"घर" ची शुद्धलेखन काय आहे?`
- `पाणी ची शुद्धलेखन काय आहे?` → `"पाणी" ची शुद्धलेखन काय आहे?`

**Examples (Assamese):**
- `ঘৰ শব্দেৰ বানান কি?` → `"ঘৰ" শব্দেৰ বানান কি?`

**Examples (Kannada):**
- `ಮನೆ ಸ್ಪೆಲಿಂಗ್ ಏನು?` → `"ಮನೆ" ಸ್ಪೆಲಿಂಗ್ ಏನು?`

**Examples (Malayalam):**
- `വീട് എങ്ങനെ എഴുതുന്നു?` → `"വീട്" എങ്ങനെ എഴുതുന്നു?`

**Examples (Odia):**
- `ଘର ଶବ୍ଦର ବାନାନ କଣ?` → `"ଘର" ଶବ୍ଦର ବାନାନ କଣ?`

**Examples (Punjabi):**
- `ਘਰ ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?` → `"ਘਰ" ਦੀ ਸਪੈਲਿੰਗ ਕੀ ਹੈ?`

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

**Version:** 4.0 (Comprehensive - 11 Indic Languages Supported)
**Supported Languages:** English, Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Assamese, Kannada, Malayalam, Odia, Punjabi