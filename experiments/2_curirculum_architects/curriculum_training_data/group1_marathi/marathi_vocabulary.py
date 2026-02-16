#!/usr/bin/env python3
"""
Marathi Vocabulary Lists for Dataset Generation
Organized by difficulty (character count) and category.
Translated from Hindi vocabulary.
"""

# Easy Words (2-4 characters)
import os
import sys

# Add parent directory to path to import from prompt_utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_expanded_vocabulary import (  # noqa: E402
    EXPANDED_ANIMALS,
    EXPANDED_BIRDS,
    EXPANDED_BODY_PARTS,
    EXPANDED_CLOTHES,
    EXPANDED_EDUCATION,
    EXPANDED_FOOD,
    EXPANDED_MISC,
    EXPANDED_NATURE,
    EXPANDED_OBJECTS,
    EXPANDED_PEOPLE,
    EXPANDED_VEGETABLES_FRUITS,
    EXPANDED_VOCABULARY_LIST,
    EXPANDED_VOCABULARY_WIKTIONARY,
)
from prompt_utils import get_marathi_grapheme_clusters  # noqa: E402

EASY_ANIMALS = [
    "कुत्रा",
    "मांजर",
    "गाय",
    "घोडा",
    "बकरी",
    "मेंढी",
    "डुकर",
    "कोंबडी",
    "उंदीर",
    "कोल्हा",
    "ससा",
    "कासव",
    "मासा",
    "पक्षी",
    "अस्वल",
    "हत्ती",
    "वाघ",
    "सिंह",
    "माकड",
    "उंट",
    "जिराफ",
    "पेंग्विन",
    "कावळा",
    "पोपट",
    "मोर",
    "बाज",
    "घुबड",
    "बदक",
    "हंस",
    "गिधाड",
]

EASY_OBJECTS = [
    "घर",
    "पाणी",
    "पुस्तक",
    "पेन",
    "दार",
    "खिडकी",
    "टेबल",
    "खुर्ची",
    "शाळा",
    "दुकान",
    "रस्ता",
    "शहर",
    "झाड",
    "फूल",
    "पान",
    "गाडी",
    "बस",
    "रेल्वे",
    "ट्रक",
    "सायकल",
    "नाव",
    "जहाज",
    "विमान",
    "संगणक",
    "फोन",
    "घड्याळ",
    "आरसा",
    "भिंत",
    "कुलूप",
    "चावी",
    "झाडू",
    "बादली",
    "दिवा",
    "पंखा",
    "पलंग",
    "सोफा",
    "अलमारी",
    "उशी",
    "चूल",
    "फ्रीज",
    "बाटली",
    "ताट",
    "टॉवेल",
    "घोंगडी",
    "चादर",
    "गादी",
]

EASY_BODY_PARTS = [
    "हात",
    "पाय",
    "डोळा",
    "कान",
    "नाक",
    "तोंड",
    "डोकं",
    "मान",
    "खांदा",
    "कोपर",
    "मनगट",
    "बोट",
    "पोट",
    "पाठ",
    "गुडघा",
    "घोटा",
    "केस",
    "दात",
    "जीभ",
    "गाल",
    "कपाळ",
    "हनुवटी",
    "घसा",
    "छाती",
    "कंबर",
]

EASY_COLORS = [
    "लाल",
    "निळा",
    "हिरवा",
    "पिवळा",
    "काळा",
    "पांढरा",
    "तपकिरी",
    "गुलाबी",
    "नारिंगी",
    "जांभळा",
    "राखाडी",
    "सोनेरी",
    "चांदी",
    "तांबे",
]

EASY_NATURE = [
    "दिवस",
    "रात्र",
    "सूर्य",
    "चंद्र",
    "तारा",
    "ढग",
    "वारा",
    "पाऊस",
    "बर्फ",
    "आग",
    "माती",
    "वाळू",
    "समुद्र",
    "नदी",
    "डोंगर",
    "जंगल",
    "वाळवंट",
    "तळे",
    "आकाश",
    "टेकडी",
    "गवत",
    "बी",
    "मूळ",
    "फांदी",
]

EASY_PEOPLE = [
    "माणूस",
    "बाई",
    "मुलगा",
    "मुलगी",
    "मूल",
    "वडील",
    "आई",
    "आजोबा",
    "आजी",
    "भाऊ",
    "बहीण",
    "मित्र",
    "लोक",
    "कुटुंब",
    "सोबती",
]

EASY_FOOD = [
    "भाकरी",
    "तांदूळ",
    "डाळ",
    "भाजी",
    "फळ",
    "दूध",
    "पाणी",
    "चहा",
    "कॉफी",
    "मीठ",
    "साखर",
    "तेल",
    "लोणी",
    "अंडे",
    "मांस",
    "मासा",
]

# Medium Words (5-6 characters)
MEDIUM_ANIMALS = [
    "हत्ती",
    "वाघ",
    "सिंह",
    "माकड",
    "ससा",
    "उंट",
    "जिराफ",
    "पेंग्विन",
    "मोर",
    "पोपट",
    "कावळा",
    "घुबड",
    "बदक",
    "हंस",
    "गिधाड",
    "उंदीर",
    "कोल्हा",
    "कासव",
    "अस्वल",
    "मासा",
    "पक्षी",
    "डुकर",
    "कोंबडी",
    "बकरी",
    "मेंढी",
]

MEDIUM_OBJECTS = [
    "विद्यालय",
    "रुग्णालय",
    "वाचनालय",
    "रेल्वेगाडी",
    "विमान",
    "संगणक",
    "दूरदर्शन",
    "फ्रीज",
    "मायक्रोवेव्ह",
    "मिक्सर",
    "केटली",
    "स्वच्छतागृह",
    "टॉवेल",
    "साबणदाणी",
    "चादर",
    "गादी",
    "अलमारी",
    "सोफा",
    "शय्या",
    "डेस्क",
    "उशी",
    "चूल",
    "फ्रीज",
    "बाटली",
    "ताट",
    "आरसा",
    "भिंत",
    "कुलूप",
    "चावी",
    "झाडू",
    "बादली",
    "दिवा",
    "दिवा",
    "पंखा",
    "घड्याळ",
    "पलंग",
    "खुर्ची",
    "टेबल",
    "दार",
    "खिडकी",
    "शाळा",
    "दुकान",
    "रस्ता",
    "शहर",
    "इमारत",
    "झाड",
    "फूल",
    "पान",
]

MEDIUM_PROFESSIONS = [
    "शिक्षक",
    "अध्यापक",
    "डॉक्टर",
    "वकील",
    "अभियंता",
    "शेतकरी",
    "दूधवाला",
    "भाजीवाला",
    "वर्तमानपत्रवाला",
    "सुतार",
    "लोहार",
    "शिंपी",
    "सोनार",
    "बांधकाम",
    "स्वयंपाकी",
    "लिपिक",
    "नाई",
    "मोची",
    "कुली",
    "कलाकार",
    "लेखक",
    "गायक",
    "नर्तक",
    "अभिनेता",
    "चित्रकार",
    "शिल्पकार",
    "शास्त्रज्ञ",
    "अभियंता",
    "वैमानिक",
    "परिचारिका",
    "पोलीस",
    "सैनिक",
    "न्यायाधीश",
    "मंत्री",
    "पंतप्रधान",
]

MEDIUM_NATURE = [
    "समुद्र",
    "नदी",
    "डोंगर",
    "जंगल",
    "वाळवंट",
    "तळे",
    "ज्वालामुखी",
    "महासागर",
    "आकाशगंगा",
    "चंद्र",
    "तारा",
    "ग्रह",
    "ढग",
    "वारा",
    "पाऊस",
    "बर्फ",
    "वादळ",
    "तुफान",
    "धुके",
    "पूर",
    "आग",
    "पाणी",
    "माती",
    "वाळू",
    "कोळसा",
    "लावा",
    "फूल",
    "गवत",
    "झाड",
    "देठ",
    "बी",
    "खोड",
    "मूळ",
    "पान",
    "फांदी",
    "तुळस",
    "कनेर",
]

MEDIUM_VEHICLES = [
    "गाडी",
    "बस",
    "रेल्वेगाडी",
    "ट्रक",
    "मोटारसायकल",
    "सायकल",
    "स्कूटर",
    "रिक्षा",
    "ट्रॅक्टर",
    "जीप",
    "जहाज",
    "नाव",
    "प्रवासीनौका",
    "नौका",
    "विमान",
    "हेलिकॉप्टर",
    "ग्लायडर",
    "ट्राम",
    "भूमिगतरेल्वे",
    "टॅक्सी",
    "बसस्टॉप",
    "रेल्वेस्टेशन",
    "विमानतळ",
    "रस्ता",
    "महामार्ग",
    "पेट्रोलपंप",
    "ट्रॅफिकलाइट",
    "कारपार्क",
]

MEDIUM_FOOD = [
    "भाकरी",
    "तांदूळ",
    "डाळ",
    "भाजी",
    "फळ",
    "दूध",
    "पाणी",
    "चहा",
    "कॉफी",
    "मीठ",
    "साखर",
    "तेल",
    "लोणी",
    "अंडे",
    "मांस",
    "मासा",
    "बटाटा",
    "टोमॅटो",
    "कांदा",
    "लसूण",
    "आले",
    "हळद",
    "कोथिंबीर",
    "जिरे",
    "मिरची",
    "गाजर",
    "मुळा",
    "वांगी",
    "भेंडी",
    "कारले",
    "दुधी",
    "तोरी",
    "भोपळा",
    "सफरचंद",
    "केळी",
    "संत्रा",
    "द्राक्षे",
    "आंबा",
    "पपई",
    "डाळिंब",
    "पेरू",
    "लिंबू",
    "काकडी",
    "खरबूज",
]

MEDIUM_HOUSEHOLD = [
    "पलंग",
    "खुर्ची",
    "टेबल",
    "सोफा",
    "शय्या",
    "अलमारी",
    "डेस्क",
    "उशी",
    "चूल",
    "फ्रीज",
    "मायक्रोवेव्ह",
    "मिक्सर",
    "केटली",
    "बाटली",
    "ताट",
    "कॅनओपनर",
    "स्वच्छतागृह",
    "आंघोळीचा टब",
    "टॉवेल",
    "आरसा",
    "साबणदाणी",
    "घोंगडी",
    "चादर",
    "गादी",
    "दार",
    "खिडकी",
    "भिंत",
    "कुलूप",
    "चावी",
    "झाडू",
    "बादली",
    "दिवा",
    "पंखा",
    "घड्याळ",
]

# Hard Words (7-9+ characters)
HARD_COMPLEX_NOUNS = [
    "विद्यालय",
    "रुग्णालय",
    "वाचनालय",
    "संगणक",
    "दूरदर्शन",
    "फ्रीज",
    "मायक्रोवेव्ह",
    "हेलिकॉप्टर",
    "भूमिगतरेल्वे",
    "रेल्वेस्टेशन",
    "विमानतळ",
    "पेट्रोलपंप",
    "ट्रॅफिकलाइट",
    "कारपार्क",
    "प्रवासीनौका",
    "मोटारसायकल",
    "अध्यापक",
    "अभियंता",
    "बांधकाम",
    "पंतप्रधान",
    "आकाशगंगा",
    "ज्वालामुखी",
    "महासागर",
    "चंद्र",
    "भूमिगतरेल्वे",
    "स्वच्छतागृह",
    "आंघोळीचा टब",
    "साबणदाणी",
    "कॅनओपनर",
    "दूधवाला",
    "वर्तमानपत्रवाला",
    "भाजीवाला",
]

HARD_ABSTRACT = [
    "आनंद",
    "दुःख",
    "प्रेम",
    "मैत्री",
    "ज्ञान",
    "विज्ञान",
    "धैर्य",
    "शांती",
    "स्वातंत्र्य",
    "समानता",
    "न्याय",
    "सत्य",
    "अहिंसा",
    "करुणा",
    "क्षमा",
    "धीर",
    "उत्साह",
    "आशा",
    "निराशा",
    "भीती",
    "हर्ष",
    "शोक",
    "अभिमान",
    "लाज",
    "राग",
    "प्रसन्नता",
    "उदासी",
    "चिंता",
    "विश्वास",
    "संशय",
]

# Days of Week
DAYS_OF_WEEK = [
    "सोमवार",
    "मंगळवार",
    "बुधवार",
    "गुरुवार",
    "शुक्रवार",
    "शनिवार",
    "रविवार",
]

# Months
MONTHS = [
    "जानेवारी",
    "फेब्रुवारी",
    "मार्च",
    "एप्रिल",
    "मे",
    "जून",
    "जुलै",
    "ऑगस्ट",
    "सप्टेंबर",
    "ऑक्टोबर",
    "नोव्हेंबर",
    "डिसेंबर",
]

# Numbers (1-100)
NUMBERS = [
    "एक",
    "दोन",
    "तीन",
    "चार",
    "पाच",
    "सहा",
    "सात",
    "आठ",
    "नऊ",
    "दहा",
    "अकरा",
    "बारा",
    "तेरा",
    "चौदा",
    "पंधरा",
    "सोळा",
    "सतरा",
    "अठरा",
    "एकोणीस",
    "वीस",
    "एकवीस",
    "बावीस",
    "तेवीस",
    "चोवीस",
    "पंचवीस",
    "सव्वीस",
    "सत्तावीस",
    "अठ्ठावीस",
    "एकोणतीस",
    "तीस",
    "एकतीस",
    "बत्तीस",
    "तेहेतीस",
    "चौतीस",
    "पस्तीस",
    "छत्तीस",
    "सदतीस",
    "अडतीस",
    "एकोणचाळीस",
    "चाळीस",
    "एकेचाळीस",
    "बेचाळीस",
    "त्रेचाळीस",
    "चव्वेचाळीस",
    "पंचेचाळीस",
    "सेहेचाळीस",
    "सत्तेचाळीस",
    "अठ्ठेचाळीस",
    "एकोणपन्नास",
    "पन्नास",
    "एक्कावन",
    "बावन्न",
    "त्रेपन्न",
    "चोपन्न",
    "पंचावन्न",
    "छप्पन्न",
    "सत्तावन्न",
    "अठ्ठावन्न",
    "एकोणसाठ",
    "साठ",
    "एकसष्ट",
    "बासष्ट",
    "त्रेसष्ट",
    "चौसष्ट",
    "पासष्ट",
    "सहासष्ट",
    "सदुसष्ट",
    "अडुसष्ट",
    "एकोणसत्तर",
    "सत्तर",
    "एकाहत्तर",
    "बाहत्तर",
    "त्र्याहत्तर",
    "चौर्‍याहत्तर",
    "पंच्याहत्तर",
    "शहात्तर",
    "सत्याहत्तर",
    "अठ्ठ्याहत्तर",
    "एकोणऐंशी",
    "ऐंशी",
    "एक्याऐंशी",
    "ब्याऐंशी",
    "त्र्याऐंशी",
    "चौर्‍याऐंशी",
    "पंच्याऐंशी",
    "शहाऐंशी",
    "सत्याऐंशी",
    "अठ्ठ्याऐंशी",
    "एकोणनव्वद",
    "नव्वद",
    "एक्याण्णव",
    "ब्याण्णव",
    "त्र्याण्णव",
    "चौऱ्याण्णव",
    "पंच्याण्णव",
    "शहाण्णव",
    "सत्याण्णव",
    "अठ्ठ्याण्णव",
    "नव्याण्णव",
    "शंभर",
]

# Rhyming Word Pairs (for Statement 5)
RHYMING_PAIRS = {
    "कमळ": "जमळ",
    "घर": "कर",
    "पाणी": "हाणी",
    "सूर्य": "पूर्य",
    "कुत्रा": "सुत्रा",
    "मांजर": "सांजर",
    "गाय": "भाय",
    "घोडा": "मोडा",
    "मूल": "कूल",
    "मित्र": "हित्र",
    "सोबती": "मोबती",
    "पुस्तक": "मुस्तक",
    "पेन": "देन",
    "टेबल": "देबल",
    "खुर्ची": "मुर्ची",
    "झाड": "माड",
    "फूल": "मूल",
    "पान": "मान",
    "हात": "सात",
    "पाय": "गाय",
    "डोळा": "मोळा",
    "कान": "मान",
    "नाक": "साक",
    "तोंड": "मोंड",
    "लाल": "माल",
    "निळा": "मिळा",
    "हिरवा": "मिरवा",
    "पिवळा": "मिवळा",
    "काळा": "माळा",
    "दिवस": "मिवस",
    "रात्र": "मात्र",
    "चंद्र": "मंद्र",
    "तारा": "मारा",
}

# Classification categories
CLASSIFICATION_CATEGORIES = {
    "प्राणी": EASY_ANIMALS + MEDIUM_ANIMALS + EXPANDED_ANIMALS + EXPANDED_BIRDS,
    "व्यक्ती": EASY_PEOPLE + MEDIUM_PROFESSIONS + EXPANDED_PEOPLE,
    "वस्तू": (
        EASY_OBJECTS
        + MEDIUM_OBJECTS
        + MEDIUM_HOUSEHOLD
        + MEDIUM_VEHICLES
        + EXPANDED_OBJECTS
        + EXPANDED_CLOTHES
        + EXPANDED_FOOD
        + EXPANDED_VEGETABLES_FRUITS
        + EXPANDED_BODY_PARTS
        + EXPANDED_EDUCATION
        + EXPANDED_MISC
        + EXPANDED_NATURE
        + EXPANDED_VOCABULARY_WIKTIONARY
    ),
}

# Combined word lists by difficulty
EASY_WORDS = (
    EASY_ANIMALS
    + EASY_OBJECTS
    + EASY_BODY_PARTS
    + EASY_COLORS
    + EASY_NATURE
    + EASY_PEOPLE
    + EASY_FOOD
)
MEDIUM_WORDS = (
    MEDIUM_ANIMALS
    + MEDIUM_OBJECTS
    + MEDIUM_PROFESSIONS
    + MEDIUM_NATURE
    + MEDIUM_VEHICLES
    + MEDIUM_FOOD
    + MEDIUM_HOUSEHOLD
)
HARD_WORDS = HARD_COMPLEX_NOUNS + HARD_ABSTRACT + DAYS_OF_WEEK + MONTHS

# All words (for general use)
ALL_WORDS = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS


# Remove duplicates while preserving order
def get_unique_words(word_list):
    """Remove duplicates while preserving order"""
    seen = set()
    unique = []
    for word in word_list:
        if word not in seen:
            seen.add(word)
            unique.append(word)
    return unique


# Classify expanded vocabulary by length (grapheme count)
expanded_easy = []
expanded_medium = []
expanded_hard = []

for word in EXPANDED_VOCABULARY_LIST:
    graphemes = get_marathi_grapheme_clusters(word)
    length = len(graphemes)
    if length <= 4:
        expanded_easy.append(word)
    elif length <= 6:
        expanded_medium.append(word)
    else:
        expanded_hard.append(word)

# Extend original lists with expanded vocabulary
EASY_WORDS = EASY_WORDS + expanded_easy
MEDIUM_WORDS = MEDIUM_WORDS + expanded_medium
HARD_WORDS = HARD_WORDS + expanded_hard
ALL_WORDS = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS


# Get unique word lists
EASY_WORDS_UNIQUE = get_unique_words(EASY_WORDS)
MEDIUM_WORDS_UNIQUE = get_unique_words(MEDIUM_WORDS)
HARD_WORDS_UNIQUE = get_unique_words(HARD_WORDS)
ALL_WORDS_UNIQUE = get_unique_words(ALL_WORDS)

# Verify word count
if __name__ == "__main__":
    print(f"Easy words (unique): {len(EASY_WORDS_UNIQUE)}")
    print(f"Medium words (unique): {len(MEDIUM_WORDS_UNIQUE)}")
    print(f"Hard words (unique): {len(HARD_WORDS_UNIQUE)}")
    print(f"Total unique words: {len(ALL_WORDS_UNIQUE)}")
    print(f"Rhyming pairs: {len(RHYMING_PAIRS)}")
    print(f"Numbers (1-100): {len(NUMBERS)}")
