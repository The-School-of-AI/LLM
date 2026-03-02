#!/usr/bin/env python3
"""
Assamese Vocabulary Lists for Dataset Generation
Expanded for 200k dataset target.
Includes: Yuktakshars, Verbs, Morphology, and Semantic relationships.
Validated against standard Assamese dictionaries (Hemkosh/Xobdo).
"""

# ==========================================
# 1. ORTHOGRAPHY & COMPLEXITY LEVELS
# ==========================================

# Easy Words (2-4 characters, simple CV patterns)
EASY_ANIMALS = [
    "কুকুৰ",
    "মেকুৰী",
    "গৰু",
    "ঘোঁৰা",
    "ছাগলী",
    "গাহৰি",
    "কুকুৰা",
    "নেউল",
    "শিয়াল",
    "সিংহ",
    "বাঘ",
    "বান্দৰ",
    "উট",
    "মাছ",
    "চৰাই",
    "হাতী",
    "ভালুক",
    "কাছ",
    "ম'হ",
    "হাঁহ",
    "পৰুৱা",
    "মকৰা",
    "পখিলা",
    "কেৰ্কেটুৱা",
    "নিগনি",
    "সাপ",
    "ভেকুলী",
    "শামুক",
]

EASY_OBJECTS = [
    "ঘৰ",
    "পানী",
    "কিতাপ",
    "কলম",
    "দুৱাৰ",
    "খিৰিকী",
    "মেজ",
    "চকী",
    "স্কুল",
    "দোকান",
    "বাট",
    "চহৰ",
    "গছ",
    "ফুল",
    "পাত",
    "গাড়ী",
    "বাছ",
    "ৰেল",
    "ট্ৰাক",
    "চাইকেল",
    "নাও",
    "জাহাজ",
    "ঘড়ী",
    "আইনা",
    "বাল্টি",
    "বিছনা",
    "কম্বল",
    "চাদৰ",
    "চাবোন",
    "গিলাচ",
    "বাটি",
    "চাকি",
    "চাবি",
    "তলা",
]

EASY_BODY_PARTS = [
    "হাত",
    "ভৰি",
    "চকু",
    "কাণ",
    "নাক",
    "মুখ",
    "মূৰ",
    "ডিঙি",
    "কান্ধ",
    "আঙুলি",
    "পেট",
    "পিঠি",
    "চুলি",
    "দাঁত",
    "জিভা",
    "গাল",
    "কঁকাল",
    "নখ",
    "ভৰিৰ পতা",
    "হাড়",
    "ছাল",
    "কপাল",
]

EASY_COLORS = [
    "ৰঙা",
    "নীলা",
    "সেউজীয়া",
    "হালধীয়া",
    "ক'লা",
    "বগা",
    "গোলাপী",
    "বেঙুনীয়া",
    "কমলা",
    "মুগা",
    "ছাই",
    "মটিয়া",
]

EASY_NATURE = [
    "দিন",
    "ৰাতি",
    "সূৰ্য",
    "জোন",
    "তৰা",
    "ডাৱৰ",
    "বতাহ",
    "বৰষুণ",
    "বৰফ",
    "নদী",
    "সমুদ্ৰ",
    "পৰ্বত",
    "জংঘল",
    "বন",
    "ঘাঁহ",
    "মাটি",
    "বালি",
    "শিল",
    "জুই",
    "ধোঁৱা",
]

EASY_PEOPLE = [
    "মানুহ",
    "ল'ৰা",
    "ছোৱালী",
    "শিশু",
    "কেঁচুৱা",
    "দেউতা",
    "মা",
    "ককা",
    "আইতা",
    "ভাই",
    "ভনী",
    "বন্ধু",
    "পৰিয়াল",
    "খুৰা",
    "মামা",
    "বাইদেউ",
    "দাদা",
    "পেহা",
    "মাহী",
]

EASY_FOOD = [
    "ভাত",
    "ৰুটি",
    "দাইল",
    "পাচলি",
    "ভাজি",
    "ফল",
    "গাখীৰ",
    "চাহ",
    "নিমখ",
    "চেনি",
    "তেল",
    "কণী",
    "মাংস",
    "পিঠা",
    "লাৰু",
    "দৈ",
    "জলকীয়া",
    "আচাৰ",
    "মৌ",
]

# Medium Words (5-6 characters, mild complexity)
MEDIUM_ANIMALS = [
    "জিৰাফ",
    "হৰিণা",
    "শহাপহু",
    "চিলনী",
    "শগুণ",
    "ঘঁৰিয়াল",
    "শিহু",
    "গঁড়",
    "নাহৰফুটুকী",
    "কুকুৰনেছীয়া",
]

MEDIUM_OBJECTS = [
    "বিদ্যালয়",
    "চিকিৎসালয়",
    "পুথিভঁৰাল",
    "কম্পিউটাৰ",
    "দূৰদৰ্শন",
    "আলমাৰী",
    "কলহ",
    "কেৰাহী",
    "কুঠাৰ",
    "দা",
    "কটাৰী",
    "বিচনা",
    "চামুচ",
    "পিয়লা",
]

MEDIUM_PROFESSIONS = [
    "শিক্ষক",
    "ডাক্তৰ",
    "অধিবক্তা",
    "কৃষক",
    "কাৰিকৰ",
    "দৰ্জী",
    "লেখক",
    "গায়ক",
    "নৃত্যশিল্পী",
    "চিত্ৰশিল্পী",
    "পুলিচ",
    "বেপাৰী",
    "মাছমৰীয়া",
    "ৰান্ধনি",
]

MEDIUM_NATURE = [
    "মহাসাগৰ",
    "চন্দ্ৰ",
    "গ্ৰহ",
    "বতৰ",
    "ঋতু",
    "বৰষুণ",
    "বিজুলী",
    "গাজনি",
    "ধুমুহা",
    "কুঁৱলী",
]

MEDIUM_VEHICLES = [
    "ৰেলগাড়ী",
    "মটৰচাইকেল",
    "ৰিক্সা",
    "উৰাজাহাজ",
    "হেলিকপ্টাৰ",
    "গৰুগাড়ী",
]

# Hard Words & Yuktakshars (Conjuncts) - CRITICAL for S1/S4
YUKTAKSHAR_WORDS = [
    "স্কুল",
    "জ্ঞান",
    "বিজ্ঞান",
    "পৰীক্ষা",
    "বিশ্বাস",
    "সন্মান",
    "উচ্চাকাংক্ষা",
    "প্ৰতিষ্ঠান",
    "ৰাষ্ট্ৰ",
    "স্বাভিমান",
    "আত্মবিশ্বাস",
    "গুৰুত্বপূৰ্ণ",
    "প্ৰযুক্তি",
    "সংস্কৃতি",
    "ঐতিহ্য",
    "লক্ষ্মী",
    "স্মৃতি",
    "প্ৰাৰ্থনা",
    "আশীৰ্বাদ",
    "প্ৰস্তুত",
    "গ্ৰন্থ",
    "শ্ৰদ্ধা",
    "ব্ৰহ্মপুত্ৰ",
    "সন্ধিয়া",
    "প্ৰকৃতি",
    "ব্যৱহাৰ",
    "সৃষ্টি",
    "দৃষ্টিভংগী",
    "পৰিস্থিতি",
    "মন্ত্ৰী",
    "যন্ত্ৰ",
    "শান্তি",
    "স্বাস্থ্য",
    "স্বতন্ত্ৰ",
    "নিৰ্দিষ্ট",
    "অস্তিত্ব",
    "গুৰুত্ব",
    "ব্যক্তিত্ব",
    "নিৰ্বাচন",
]

# Words specific to Assamese nuances (Wa vs Ba, Sibilants)
ASSAMESE_SPECIFIC_PHONETIC = [
    "কুঁৱা",
    "গুৱাহাটী",
    "দেৱাল",
    "সেৱা",
    "খোৱা",
    "শোৱা",
    "যোৱা",  # 'Wa' emphasis
    "বিশেষ",
    "শেষ",
    "আকাশ",
    "বাৰিষা",
    "ভাষা",
    "বিশ্বাস",  # 'Sha' variations
    "গামোচা",
    "জাপি",
    "সৰাই",
    "মুগা",
    "এৰী",
    "পাট",  # Cultural items
    "নামঘৰ",
    "সত্ৰ",
    "ভাওনা",
    "বিহু",
]

HARD_ABSTRACT = [
    "স্বাধীনতা",
    "গণতন্ত্ৰ",
    "সংবিধান",
    "অৰ্থনীতি",
    "পৰিৱেশ",
    "আন্দোলন",
    "সাহিত্য",
    "উপন্যাস",
    "কাব্য",
    "দৰ্শন",
    "মনোবিজ্ঞান",
    "সামাজিক",
    "ৰাজনৈতিক",
    "ঐতিহাসিক",
    "ভৌগোলিক",
    "প্ৰশাসনিক",
    "কাৰ্যালয়",
    "নাগৰিকত্ব",
    "অধিকাৰ",
    "বিৱৰ্তন",
    "প্ৰযুক্তিবিদ্যা",
]

# ==========================================
# 2. GRAMMAR & MORPHOLOGY (For S11)
# ==========================================

# Common Verbs (Roots & Conjugations)
VERBS = [
    "খোৱা",
    "যোৱা",
    "কৰা",
    "পঢ়া",
    "লিখা",
    "শুনা",
    "দেখা",
    "হাঁহা",
    "কন্দা",
    "খেলা",
    "দৌৰা",
    "মৰা",
    "অনা",
    "লোৱা",
    "দিয়া",
    "পঠিওৱা",
    "বহা",
    "উঠা",
    "শোৱা",
    "ৰন্ধা",
    "চলোৱা",
    "ভবা",
    "নচা",
    "জনা",
    "ধৰা",
    "ক্লান্ত হোৱা",
    "শিকোৱা",
    "বিচৰা",
    "ৰখা",
    "পৰা",
]

# Adjectives
ADJECTIVES = [
    "ধুনীয়া",
    "বেয়া",
    "ভাল",
    "ডাঙৰ",
    "সৰু",
    "ওখ",
    "চুটি",
    "শকত",
    "ক্ষীণ",
    "ৰঙা",
    "মিঠা",
    "টেঙা",
    "তিতা",
    "গৰম",
    "ঠাণ্ডা",
    "কোমল",
    "টান",
    "নতুন",
    "পুৰণি",
    "দুখীয়া",
    "ধনী",
]

# Assamese Suffixes (Bibhakti) for Morphology Tasks
SUFFIXES = {
    "plural": ["বোৰ", "বিলাক", "সকল", "হঁত", "মখা"],
    "definite": [
        "টো",
        "টি",
        "জন",
        "জনী",
        "খন",
        "খনি",
        "ডাল",
        "চটা",
        "গছ",
        "পাত",
        "জোপা",
    ],
    "case": ["ৰ", "লৈ", "ত", "ৰে", "ৰপৰা", "লৈকে", "ক"],
}

# Onomatopoeic Words (Anukaran Shabda)
ONOMATOPOEIC = [
    "ৰিমঝিম",
    "ধকধক",
    "খৰখৰ",
    "মৰমৰ",
    "তিৰবিৰ",
    "জলমল",
    "কলকল",
    "টোপটোপ",
    "লৰচৰ",
    "ধুমধাম",
    "গুনগুন",
    "ঘৰঘৰ",
    "চিকমিক",
]

# Idioms (Khandabakya)
# IDIOMS = [
#     "কঁকাল ভগা", "কপাল ফুলা", "মুখ ফুলা", "আকাশত চাং পতা",
#     "পানীৰ মিঠৈ", "বালিৰ বান্ধ", "পেটত কথা ৰখা", "কাণ সৰা"
# ]

# ==========================================
# 3. SEMANTICS (For S12)
# ==========================================

SYNONYMS = {
    "পানী": ["জল", "নীৰ", "সলিল"],
    "ঘৰ": ["গৃহ", "ভৱন", "আৱাস", "নিৱাস", "আলয়"],
    "আকাশ": ["গগন", "অম্বৰ"],
    "পৃথিৱী": ["ধৰা", "ধৰণী", "বসুন্ধৰা"],
    "দিন": ["দিৱস", "দিবা"],
    "ৰাতি": ["নিশা", "ৰজনী"],
    "চকু": ["নয়ন", "নেত্ৰ", "অক্ষি", "লোচন"],
    "গছ": ["বৃক্ষ", "তৰু"],
    "পাহাৰ": ["পৰ্বত", "গিৰি"],
    "নদী": ["নৈ", "তটিনী"],
    "মাক": ["আই", "মাতৃ", "মা"],
    "দেউতা": ["পিতৃ", "পিতাই", "পিতা", "বোপাই"],
    "ফুল": ["পুষ্প", "কুসুম"],
    "সোণ": ["স্বৰ্ণ", "কাঞ্চন"],
    "মানুহ": ["নৰ", "মনুষ্য", "মানৱ"],
}

ANTONYMS = {
    "দিন": "ৰাতি",
    "ভাল": "বেয়া",
    "সঁচা": "মিছা",
    "জন্ম": "মৃত্যু",
    "জীৱন": "মৰণ",
    "পুৱা": "গধূলি",
    "আৰম্ভ": "শেষ",
    "নতুন": "পুৰণি",
    "ডাঙৰ": "সৰু",
    "ওখ": "চুটি",
    "গৰম": "ঠাণ্ডা",
    "সুখ": "দুখ",
    "পোহৰ": "আন্ধাৰ",
    "বন্ধু": "শত্ৰু",
    "ধনী": "দুখীয়া",
    "স্বৰ্গ": "নৰক",
    "লাভ": "লোকচান",
    "জয়": "পৰাজয়",
    "ন্যায়": "অন্যায়",
    "স্বাধীন": "পৰাধীন",
}

# ==========================================
# 4. REFERENCE LISTS
# ==========================================

# Days of Week
DAYS_OF_WEEK = [
    "সোমবাৰ",
    "মঙ্গলবাৰ",
    "বুধবাৰ",
    "বৃহস্পতিবাৰ",
    "শুক্রবাৰ",
    "শনিবাৰ",
    "ৰবিবাৰ",
]

# Months
MONTHS = [
    "জানুৱাৰী",
    "ফেব্ৰুৱাৰী",
    "মাৰ্চ",
    "এপ্ৰিল",
    "মে’",
    "জুন",
    "জুলাই",
    "আগষ্ট",
    "ছেপ্তেম্বৰ",
    "অক্টোবৰ",
    "নৱেম্বৰ",
    "ডিচেম্বৰ",
    "বহাগ",
    "জেঠ",
    "আহাৰ",
    "শাওণ",
    "ভাদ",
    "আহিন",
    "কাতি",
    "আঘোণ",
    "পুহ",
    "মাঘ",
    "ফাগুন",
    "চ’ত",
]

# Basic Numbers (Used as seeds for the generator)
NUMBERS_BASE = [
    "এক",
    "দুই",
    "তিনি",
    "চাৰি",
    "পাঁচ",
    "ছয়",
    "সাত",
    "আঠ",
    "ন",
    "দহ",
    "এঘাৰ",
    "বাৰ",
    "তেৰ",
    "চৈধ্য",
    "পোন্ধৰ",
    "ষোল্ল",
    "সোতৰ",
    "ওঠৰ",
    "ঊনৈছ",
    "বিছ",
    "ত্ৰিছ",
    "চল্লিছ",
    "পঞ্চাছ",
    "ষাঠি",
    "সত্তৰ",
    "আশী",
    "নব্বৈ",
    "এশ",
    "হাজাৰ",
    "লাখ",
    "কোটি",
]

ORDINALS = [
    "প্ৰথম",
    "দ্বিতীয়",
    "তৃতীয়",
    "চতুৰ্থ",
    "পঞ্চম",
    "ষষ্ঠ",
    "সপ্তম",
    "অষ্টম",
    "নৱম",
    "দশম",
]

# Rhyming Groups (Phonetic endings)
RHYMING_GROUPS = {
    "aa": ["মা", "চা", "খা", "যা", "পা"],
    "aar": ["আকাৰ", "বজাৰ", "হাজাৰ", "আচাৰ", "বিচাৰ"],
    "on": ["জীৱন", "মৰণ", "সপোন", "আপোন", "গোপন", "পৱন"],
    "aati": ["ৰাতি", "মাটি", "বাটি", "ঘাঁটি"],
    "or": ["ঘৰ", "কৰ", "ধৰ", "বৰ", "চৰ"],
    "aan": ["গান", "ধান", "পান", "মান", "দান", "স্নান"],
    "ee": ["পানী", "ৰাণী", "বাণী", "ধনী"],
    "i": ["হাঁহি", "হাতি", "বালি", "ৰাতি"],
}

# Rhyming Word Pairs (Legacy support for S5)
RHYMING_PAIRS = {
    # -aa / -al
    "জাল": "শাল",  # Net : Shawl/Loom
    "আই": "ভাই",  # Mother : Brother
    # -aan
    "মান": "দান",  # Respect : Donation
    "গান": "প্ৰাণ",  # Song : Life
    "ধান": "বাণ",  # Paddy : Arrow/Flood
    # -at
    "হাত": "সাত",  # Hand : Seven
    "ভাত": "গাত",  # Rice : Body/Hole
    "জাত": "মাত",  # Caste : Voice
    # -or
    "ঘৰ": "কৰ",  # House : Tax/Do
    "লৰ": "চৰ",  # Run : Slap
    "বৰ": "দৰ",  # Groom/Big : Price
    "শৰ": "পৰ",  # Arrow : Fall
    # -on
    "মন": "বন",  # Mind : Forest
    "জন": "ধন",  # Person : Wealth
    "গণ": "ৰণ",  # People : War
    "সোণ": "কোণ",  # Gold : Corner
    # -ul / -uli
    "ফুল": "মূল",  # Flower : Root
    "ভুল": "কূল",  # Mistake : Shore
    "চুলি": "বুলি",  # Hair : Saying
    "ধূলি": "বালি",  # Dust : Sand
    # -i / -ti
    "ৰাতি": "বাতি",  # Night : Bowl
    "মাটি": "কাটি",  # Soil : Cut
    "লাঠি": "গাঁঠি",  # Stick : Knot
    "ধনী": "মণি",  # Rich : Jewel
    # -oka / -oku / -ukh
    "দুখ": "সুখ",  # Sadness : Happiness
    "চকু": "বুকু",  # Eye : Chest
    "নাক": "শাক",  # Nose : Vegetable
    "ককা": "পকা",  # Grandpa : Ripe
    "বোকা": "চোকা",  # Mud : Sharp
    "টকা": "চকা",  # Money : Wheel
}

# Classification categories (Expanded)
CLASSIFICATION_CATEGORIES = {
    "জীৱ (Living)": EASY_ANIMALS + MEDIUM_ANIMALS,
    "ব্যক্তি (People)": EASY_PEOPLE + MEDIUM_PROFESSIONS,
    "বস্তু (Objects)": EASY_OBJECTS + MEDIUM_OBJECTS + MEDIUM_VEHICLES,
    "খাদ্য (Food)": EASY_FOOD,
    "প্ৰকৃতি (Nature)": EASY_NATURE + MEDIUM_NATURE,
    "কৰ্ম (Action)": VERBS,
    "গুণ (Quality)": ADJECTIVES,
}

# ==========================================
# 5. AGGREGATION
# ==========================================

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
    + VERBS
    + ADJECTIVES
)

HARD_WORDS = (
    YUKTAKSHAR_WORDS
    + HARD_ABSTRACT
    + DAYS_OF_WEEK
    + MONTHS
    + ASSAMESE_SPECIFIC_PHONETIC
    + ONOMATOPOEIC
    # + [i.replace(" ", "_") for i in IDIOMS]  # Uncomment when IDIOMS is enabled
)

# All words (for general use)
ALL_WORDS = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS


def get_unique_words(word_list):
    """Remove duplicates while preserving order"""
    seen = set()
    unique = []
    for word in word_list:
        if word not in seen:
            seen.add(word)
            unique.append(word)
    return unique


# Get unique word lists
EASY_WORDS_UNIQUE = get_unique_words(EASY_WORDS)
MEDIUM_WORDS_UNIQUE = get_unique_words(MEDIUM_WORDS)
HARD_WORDS_UNIQUE = get_unique_words(HARD_WORDS)
ALL_WORDS_UNIQUE = get_unique_words(ALL_WORDS)
YUKTAKSHAR_UNIQUE = get_unique_words(YUKTAKSHAR_WORDS)

if __name__ == "__main__":
    print(f"Easy words: {len(EASY_WORDS_UNIQUE)}")
    print(f"Medium words: {len(MEDIUM_WORDS_UNIQUE)}")
    print(f"Hard/Complex words: {len(HARD_WORDS_UNIQUE)}")
    print(f"Yuktakshar words: {len(YUKTAKSHAR_UNIQUE)}")
    print(f"Total unique words: {len(ALL_WORDS_UNIQUE)}")
    print(f"Synonym Pairs: {len(SYNONYMS)}")
    print(f"Antonym Pairs: {len(ANTONYMS)}")
    print(f"Suffixes: {sum(len(v) for v in SUFFIXES.values())}")
