#!/usr/bin/env python3
"""
Generate Group 3 Shapes, Colors & Patterns Dataset (250,000 samples)
Creates 4 statement types with 35 sub-generators total.
"""

import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import combine_qa_pairs_to_reach_min_tokens  # noqa: E402


# Post-processing for prompt style consistency (Group 3)
def _finalize_group3_prompt(p: str) -> str:
    p = p.strip()
    # Remove ellipses if any slipped in
    p = p.replace("...", "").replace("…", "")

    # Prefer simple textbook phrasing
    p = re.sub(
        r"^Do you know what color (.+?) is\??$",
        r"What color is \1?",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"^Can you tell me what color (.+?) is\??$",
        r"What color is \1?",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"^Tell me what color (.+?) is\??$",
        r"What color is \1?",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"^What is the color of (.+?)\??$", r"What color is \1?", p, flags=re.IGNORECASE
    )

    # Fix patterns like "Compare X and Y. Which has..." -> "Compare X and Y, which has..."
    # Fix patterns like "You have X. If..." -> "You have X, if..."
    # Fix patterns like "Add X. What's..." -> "Add X, what's..."

    # Replace ". If" with ", if"
    p = re.sub(r"\.\s+If\s", r", if ", p)
    # Replace ". How" with ", how" (when followed by question word)
    p = re.sub(
        r"\.\s+How\s+(many|much|do|does|is|are|can)",
        r", how \1",
        p,
        flags=re.IGNORECASE,
    )
    # Replace ". What" with ", what" (handles both "What " and "What's", "What's", etc.)
    p = re.sub(
        r"\.\s+What(\'s|\'s| is| do| does| can|\s)", r", what\1", p, flags=re.IGNORECASE
    )
    # Replace ". Which" with ", which"
    p = re.sub(r"\.\s+Which\s", r", which ", p)
    # Replace ". Tell" with ", tell" (when it's "tell me")
    p = re.sub(r"\.\s+Tell\s+me\s", r", tell me ", p, flags=re.IGNORECASE)
    # Replace ". Observe" with ", observe"
    p = re.sub(r"\.\s+Observe\s", r", observe ", p, flags=re.IGNORECASE)

    # Ensure uniform question-mark ending
    p = p.rstrip()
    if not p.endswith("?"):
        p = p.rstrip(".!")
        p = p + "?"
    return p


# ============================================================================
# DATA MODULES
# ============================================================================

# Color-Object Mappings (based on prompt lines 43-54)
COLOR_OBJECTS = {
    "red": [
        "apple",
        "tomato",
        "strawberry",
        "cherry",
        "fire truck",
        "stop sign",
        "rose",
        "ladybug",
        "cardinal",
        "blood",
        "brick",
        "chili pepper",
        "lobster",
        "crab",
        "pomegranate",
        "raspberry",
        "mars",
        "ruby",
        "poppy",
        "heart",
        "fire hydrant",
        "barn",
        "robin breast",
    ],
    "orange": [
        "orange",
        "carrot",
        "pumpkin",
        "goldfish",
        "tiger",
        "basketball",
        "traffic cone",
        "autumn leaves",
        "marigold",
        "papaya",
        "sweet potato",
        "tangerine",
        "apricot",
        "peach",
        "cantaloupe",
        "salmon",
        "fox",
        "monarch butterfly",
    ],
    "yellow": [
        "banana",
        "sun",
        "lemon",
        "school bus",
        "taxi",
        "sunflower",
        "corn",
        "butter",
        "daffodil",
        "canary",
        "rubber duck",
        "bee",
        "honey",
        "mustard",
        "egg yolk",
        "pineapple",
        "gold",
        "lightning",
        "tennis ball",
        "highlighter",
        "chick",
        "giraffe",
    ],  # Mango removed (can be yellow/green depending on ripeness)
    "green": [
        "grass",
        "leaf",
        "frog",
        "broccoli",
        "cucumber",
        "lime",
        "peas",
        "lettuce",
        "spinach",
        "kale",
        "avocado",
        "kiwi",
        "green apple",
        "pickle",
        "shamrock",
        "alligator",
        "crocodile",
        "turtle",
        "parsley",
        "mint",
        "basil",
        "dollar bill",
        "emerald",
        "green pepper",
        "celery",
        "cabbage",
        "asparagus",
        "zucchini",
    ],
    "blue": [
        "sky",
        "ocean",
        "water",
        "blueberry",
        "blue jay",
        "whale",
        "dolphin",
        "sapphire",
        "jeans",
        "denim",
        "peacock feathers",
        "hydrangea",
        "cornflower",
        "bluebell",
        "bluebonnet",
        "swimming pool",
        "globe",
        "ink",
        "navy uniform",
    ],
    # Note: "police car lights" removed - they are red and blue, not just blue
    "purple": [
        "grape",
        "eggplant",
        "plum",
        "lavender",
        "violet",
        "amethyst",
        "lilac",
        "iris",
        "orchid",
        "purple cabbage",
        "beet",
        "fig",
        "blackberry",
        "bruise",
        "purple onion",
        "purple potato",
        "wisteria",
    ],
    "pink": [
        "flamingo",
        "pig",
        "cotton candy",
        "rose",
        "cherry blossom",
        "peony",
        "carnation",
        "bubblegum",
        "piglet",
        "salmon",
        "ham",
        "shrimp",
        "tongue",
        "lips",
        "grapefruit",
        "dragon fruit",
    ],
    "brown": [
        "chocolate",
        "tree trunk",
        "wood",
        "dirt",
        "soil",
        "mud",
        "coffee",
        "bear",
        "horse",
        "deer",
        "acorn",
        "walnut",
        "coconut shell",
        "bread crust",
        "cookie",
        "potato skin",
        "mushroom",
        "cardboard",
        "leather",
        "football",
        "pinecone",
        "owl",
        "sparrow",
        "camel",
        "peanut",
    ],
    "black": [
        "coal",
        "crow",
        "raven",
        "panther",
        "tire",
        "night sky",
        "licorice",
        "blackberry",
        "blackbird",
        "gorilla",
        "bat",
        "spider",
        "ant",
        "orca",
        "tuxedo",
        "piano keys",
        "chalkboard",
        "shadow",
        "penguin",
    ],
    "white": [
        "snow",
        "cloud",
        "milk",
        "cotton",
        "polar bear",
        "swan",
        "dove",
        "egg",
        "rice",
        "salt",
        "sugar",
        "vanilla ice cream",
        "paper",
        "sheep",
        "rabbit",
        "ghost",
        "wedding dress",
        "teeth",
        "pearl",
        "daisy",
        "seagull",
        "marshmallow",
        "coconut flesh",
        "garlic",
        "cauliflower",
        "onion",
    ],
    # Note: "wool" removed - uncountable noun, shouldn't use "a wool"
    "gray": [
        "elephant",
        "dolphin",
        "shark",
        "mouse",
        "rat",
        "rhinoceros",
        "hippopotamus",
        "wolf",
        "storm cloud",
        "concrete",
        "stone",
        "rock",
        "silver",
        "pencil lead",
        "ash",
        "smoke",
        "seal",
        "pigeon",
        "donkey",
        "brain",
        "dust",
    ],
}

# All colors as a list (basic colors first for curriculum progression)
ALL_COLORS = [
    "red",
    "yellow",
    "blue",
    "green",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
]

# ============================================================================
# PARAMETRIC VARIATION DATA (for expanding combinatorial space)
# ============================================================================

# Adjectives for objects
ADJECTIVES = [
    # Keep only simple, universally applicable size adjectives.
    # Avoid words like typical/regular/normal/average/lovely which create unnatural phrasing.
    "big",
    "small",
    "large",
    "tiny",
]

# Contexts/Locations where objects might be found
CONTEXTS = [
    "in nature",
    "in the wild",
    "in a garden",
    "in a kitchen",
    "in a forest",
    "at the zoo",
    "at the farm",
    "at the beach",
    "in the ocean",
    "in a park",
    "in a store",
    "at home",
    "in a classroom",
    "in a museum",
    "outside",
    "in a field",
    "by the river",
    "in the mountains",
    "at the market",
]

# Time/State modifiers
STATES = [
    "usually",
    "typically",
    "normally",
    "generally",
    "commonly",
    "most often",
    "in general",
    "as a rule",
    "by default",
]

# Question starters for variety
QUESTION_STARTERS = [
    "Can you tell me",
    "Do you know",
    "I wonder",
    "Please tell me",
    "I'd like to know",
    "Could you say",
    "What would you say",
]

# Flatten all objects
ALL_COLOR_OBJECTS = []
for color_objs in COLOR_OBJECTS.values():
    ALL_COLOR_OBJECTS.extend(color_objs)

# Shape-Object Mappings (based on prompt lines 205-224)
SHAPE_OBJECTS_2D = {
    "circle": [
        "pizza",
        "wheel",
        "clock face",
        "coin",
        "plate",
        "frisbee",
        "donut",
        "moon",
        "sun",
        "CD",
        "pancake",
        "cookie",
        "orange slice",
        "basketball",
        "button",
        "manhole cover",
        "pie",
        "ring",
        "zero",
    ],
    "square": [
        "window",
        "chessboard square",
        "tile",
        "cracker",
        "napkin",
        "sticky note",
        "picture frame",
        "Rubik's cube face",
        "waffle",
        "sandwich",
        "dice face",
        "pixel",
        "coaster",
        "ice cube",
    ],
    "rectangle": [
        "door",
        "book",
        "phone",
        "TV screen",
        "laptop screen",
        "brick",
        "dollar bill",
        "envelope",
        "business card",
        "chocolate bar",
        "ruler",
        "notebook",
        "flag",
        "window pane",
        "desk",
        "table",
        "pool table",
        "mattress",
    ],
    "triangle": [
        "pizza slice",
        "sail",
        "yield sign",
        "pyramid face",
        "mountain peak",
        "tent",
        "arrow tip",
        "hanger",
        "slice of cake",
        "nachos chip",
        "sandwich",
        "roof",
        "ice cream cone",
    ],
    "oval": [
        "egg",
        "rugby ball",
        "face shape",
        "mirror",
        "track",
        "racetrack",
        "bathtub",
        "leaf",
        "grape",
        "platter",
        "spoon",
        "eye shape",
        "zero",
    ],
    "diamond": [
        "kite",
        "playing card diamond",
        "baseball field",
        "argyle pattern",
        "gemstone",
        "road sign",
    ],
    "pentagon": ["home plate", "road sign", "star fruit"],
    "hexagon": ["honeycomb cell", "nut", "benzene ring", "snowflake base"],
    "octagon": ["stop sign", "tile", "umbrella"],
    "star": ["star shape", "starfish", "sheriff badge", "Christmas star"],
}

SHAPE_OBJECTS_3D = {
    "sphere": [
        "ball",
        "globe",
        "marble",
        "orange",
        "grape",
        "bubble",
        "pearl",
        "planet",
        "moon",
        "basketball",
        "tennis ball",
        "golf ball",
        "ping pong ball",
        "balloon",
        "eyeball",
        "cherry",
        "blueberry",
        "plum",
        "ornament",
    ],
    "cube": [
        "dice",
        "ice cube",
        "Rubik's cube",
        "sugar cube",
        "box",
        "building block",
        "gift box",
    ],
    "cylinder": [
        "can",
        "battery",
        "candle",
        "tube",
        "pipe",
        "log",
        "pillar",
        "column",
        "barrel",
        "drinking glass",
        "roll of tape",
        "crayon",
        "marker",
        "pencil",
        "lipstick",
        "test tube",
    ],
    "cone": [
        "ice cream cone",
        "traffic cone",
        "party hat",
        "funnel",
        "megaphone",
        "pine tree",
        "tornado",
        "volcano",
        "teepee",
        "rocket nose",
    ],
    "rectangular prism": [
        "book",
        "brick",
        "box",
        "refrigerator",
        "shoe box",
        "cereal box",
        "door",
        "building",
        "eraser",
        "smartphone",
        "tissue box",
        "mattress",
        "loaf of bread",
    ],
    "pyramid": ["Egyptian pyramid", "tent", "roof", "mountain", "tetrahedron"],
    "oval": ["egg", "avocado", "lemon", "potato", "football"],
}

# All 2D and 3D shapes
ALL_SHAPES_2D = list(SHAPE_OBJECTS_2D.keys())
ALL_SHAPES_3D = list(SHAPE_OBJECTS_3D.keys())

# Flatten all shape objects
ALL_SHAPE_OBJECTS = []
for shape_objs in SHAPE_OBJECTS_2D.values():
    ALL_SHAPE_OBJECTS.extend(shape_objs)
for shape_objs in SHAPE_OBJECTS_3D.values():
    ALL_SHAPE_OBJECTS.extend(shape_objs)

# Geometric Properties (based on prompt lines 357-580)
# Polygon data (up to hard level - decagon/10 sides max)
SIDES_TO_SHAPE = {
    3: "triangle",
    4: "quadrilateral",
    5: "pentagon",
    6: "hexagon",
    7: "heptagon",
    8: "octagon",
    9: "nonagon",
    10: "decagon",
}

SHAPE_TO_SIDES = {
    # Basic triangles
    "triangle": 3,
    "equilateral triangle": 3,
    "isosceles triangle": 3,
    "scalene triangle": 3,
    "right triangle": 3,
    # Quadrilaterals
    "square": 4,
    "rectangle": 4,
    "parallelogram": 4,
    "rhombus": 4,
    "trapezoid": 4,
    "kite": 4,
    "quadrilateral": 4,
    # Higher polygons (up to decagon)
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
    "nonagon": 9,
    "decagon": 10,
}

SHAPE_3D_PROPERTIES = {
    # Prisms
    "cube": {"faces": 6, "edges": 12, "vertices": 8},
    "rectangular prism": {"faces": 6, "edges": 12, "vertices": 8},
    "triangular prism": {"faces": 5, "edges": 9, "vertices": 6},
    "hexagonal prism": {"faces": 8, "edges": 18, "vertices": 12},
    # Pyramids
    "square pyramid": {"faces": 5, "edges": 8, "vertices": 5},
    "triangular pyramid": {"faces": 4, "edges": 6, "vertices": 4},
    # Round shapes
    "cylinder": {"faces": 3, "edges": 2, "vertices": 0},
    "cone": {"faces": 2, "edges": 1, "vertices": 1},
    "sphere": {"faces": 1, "edges": 0, "vertices": 0},
}

SHAPE_SYMMETRY = {
    # Round shapes
    "circle": "infinite",
    # Triangles
    "equilateral triangle": 3,
    "isosceles triangle": 1,
    # Quadrilaterals
    "square": 4,
    "rectangle": 2,
    "parallelogram": 2,
    "rhombus": 2,
    # Higher polygons
    "pentagon": 5,
    "hexagon": 6,
    "octagon": 8,
}

# Color mixing rules (expanded for more combinations)
COLOR_MIXING = {
    # Primary color mixing
    ("red", "blue"): "purple",
    ("blue", "red"): "purple",
    ("red", "yellow"): "orange",
    ("yellow", "red"): "orange",
    ("blue", "yellow"): "green",
    ("yellow", "blue"): "green",
    # Tints (with white)
    ("red", "white"): "pink",
    ("white", "red"): "pink",
    ("blue", "white"): "light blue",
    ("white", "blue"): "light blue",
    ("green", "white"): "light green",
    ("white", "green"): "light green",
    ("yellow", "white"): "cream",
    ("white", "yellow"): "cream",
    ("orange", "white"): "peach",
    ("white", "orange"): "peach",
    ("purple", "white"): "lavender",
    ("white", "purple"): "lavender",
    # Shades (with black)
    ("black", "white"): "gray",
    ("white", "black"): "gray",
    ("red", "black"): "maroon",
    ("black", "red"): "maroon",
    ("blue", "black"): "navy",
    ("black", "blue"): "navy",
    ("green", "black"): "dark green",
    ("black", "green"): "dark green",
    ("yellow", "black"): "olive",
    ("black", "yellow"): "olive",
    ("orange", "black"): "brown",
    ("black", "orange"): "brown",
    ("purple", "black"): "dark purple",
    ("black", "purple"): "dark purple",
    # Secondary mixing
    ("orange", "green"): "brown",
    ("green", "orange"): "brown",
    ("purple", "orange"): "brown",
    ("orange", "purple"): "brown",
    ("purple", "green"): "brown",
    ("green", "purple"): "brown",
    # Tertiary colors
    ("red", "orange"): "red-orange",
    ("orange", "red"): "red-orange",
    ("yellow", "orange"): "yellow-orange",
    ("orange", "yellow"): "yellow-orange",
    ("yellow", "green"): "yellow-green",
    ("green", "yellow"): "yellow-green",
    ("blue", "green"): "teal",
    ("green", "blue"): "teal",
    ("blue", "purple"): "indigo",
    ("purple", "blue"): "indigo",
    ("red", "purple"): "magenta",
    ("purple", "red"): "magenta",
}

# Color associations (prompt lines 165-177) - keeping original quality data
COLOR_ASSOCIATIONS = {
    "red": ["stop", "danger", "love", "anger", "heat", "fire", "emergency", "warning"],
    "green": ["go", "nature", "growth", "environment", "safety", "money"],
    "yellow": ["caution", "happiness", "sunshine", "warning", "energy"],
    "blue": ["sadness", "calm", "trust", "cold", "water", "sky", "peace"],
    "white": ["purity", "peace", "cleanliness", "innocence", "snow"],
    "black": ["death", "darkness", "elegance", "mystery", "night", "formal"],
    "orange": ["energy", "enthusiasm", "warmth", "autumn", "creativity"],
    "purple": ["royalty", "luxury", "mystery", "magic", "spirituality"],
    "pink": ["love", "femininity", "sweetness", "romance", "tenderness"],
    "brown": ["earth", "stability", "reliability", "nature", "comfort"],
    "gray": ["neutrality", "balance", "maturity", "wisdom"],
}

# 2D-3D relationships (up to hard level)
SHAPE_2D_3D = {
    # Basic 3D to 2D face relationships
    "sphere": "circle",
    "cube": "square",
    "cylinder": "circle",
    "cone": "circle",
    "pyramid": "triangle",
    "rectangular prism": "rectangle",
    # Hard level
    "triangular prism": "triangle",
    "hexagonal prism": "hexagon",
    "triangular pyramid": "triangle",
    "square pyramid": "square",
    "hemisphere": "circle",
}

# Circle properties (prompt lines 564-573)
CIRCLE_PROPERTIES = {
    "distance around": "circumference",
    "distance across through center": "diameter",
    "half of diameter": "radius",
    "distance from center to edge": "radius",
    "perimeter": "circumference",
    "line across through center": "diameter",
}

# Pattern items (prompt lines 632-807)
PATTERN_ITEMS = {
    "colors": [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
    ],
    "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "shapes": ["circle", "square", "triangle", "star"],
    "sizes": ["big", "small", "medium"],
    "numbers": list(range(1, 101)),
    "days": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    "months": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "words": ["yes", "no", "up", "down", "left", "right"],
}

# ============================================================================
# TEMPLATES
# ============================================================================

# Statement 1: Color Perception Templates

# 1A: Object Color Identification (expanded from 10 to 40+ templates)
TEMPLATES_1A = [
    # Natural, book-like questions only
    "What color is the {obj}?",
    "What color is {obj}?",
    "What color are {obj_plural}?",
    "What color does the {obj} have?",
    "What color does {obj} have?",
]

# 1A with adjectives (REMOVED - creates unnatural phrasing)
# Only use simple size adjectives that make sense
TEMPLATES_1A_ADJ = [
    "What color is a big {obj}?",
    "What color is a small {obj}?",
    "What color is a large {obj}?",
]

# 1A with context (REMOVED - creates unnatural phrasing)
TEMPLATES_1A_CONTEXT = []

# 1B: Reverse Color Identification (expanded from 10 to 40+ templates)
TEMPLATES_1B = [
    # Natural, book-like questions only
    "What is something that is {color}?",
    "What is something that is {color}?",
    # Avoid "Name a {color} object" which can create "a orange"
    "What object is {color}?",
    "Can you name something that is {color}?",
    "What can you think of that is {color}?",
    "What has the color {color}?",
]

# 1B with context (REMOVED - creates unnatural phrasing)
TEMPLATES_1B_CONTEXT = []

# 1C: Color Verification (expanded from 5 to 30+ templates)
TEMPLATES_1C = [
    # Basic questions
    "Is a {obj} {color}?",
    "Is the {obj} {color}?",
    "Does a {obj} have {color} color?",
    "Is {color} the color of a {obj}?",
    "Would you say a {obj} is {color}?",
    # Extended variations
    "Is it true that a {obj} is {color}?",
    "Can a {obj} be {color}?",
    "Is a {obj} typically {color}?",
    "Are {obj_plural} {color}?",
    "Do {obj_plural} have a {color} color?",
    "Is {color} a common color for a {obj}?",
    "Would a {obj} be considered {color}?",
    "Is a {obj} known to be {color}?",
    "Would you describe a {obj} as {color}?",
    "Is it correct that a {obj} is {color}?",
    "True or false: a {obj} is {color}",
    "Is a {obj} usually {color}?",
    "Is the color of a {obj} {color}?",
    "Could you say a {obj} is {color}?",
    "Do people consider a {obj} {color}?",
    "Is a {obj} generally {color}?",
    "Would a {obj} appear {color}?",
    "Is a {obj} naturally {color}?",
    "Is {color} what color a {obj} is?",
    "A {obj} is {color}, right?",
    "Can we say a {obj} is {color}?",
    "Is it accurate that a {obj} is {color}?",
    "Would it be correct to call a {obj} {color}?",
    # Avoid "typical" phrasing; keep it simple
    "Is the {obj} {color}?",
    "Does a {obj} come in {color}?",
]

# 1C with adjectives (parametric expansion)
TEMPLATES_1C_ADJ = [
    "Is a {adj} {obj} {color}?",
    "Would a {adj} {obj} be {color}?",
    "Does a {adj} {obj} have {color} color?",
    "Is {color} the color of a {adj} {obj}?",
]

# 1D: Color Multiple Choice (expanded from 5 to 30+ templates)
TEMPLATES_1D = [
    # Basic questions
    "Which is {color}, a {obj1} or a {obj2}?",
    "Which one is {color}: {obj1} or {obj2}?",
    "Pick the {color} object: {obj1} or {obj2}",
    "Between {obj1} and {obj2}, which is {color}?",
    "Is {obj1} or {obj2} {color}?",
    # Extended variations
    "Which of these is {color}: {obj1} or {obj2}?",
    "Select the {color} item: {obj1} or {obj2}?",
    "Which has {color} color, {obj1} or {obj2}?",
    "Out of {obj1} and {obj2}, which is {color}?",
    "Choose the {color} one: {obj1} or {obj2}",
    "What is {color}, {obj1} or {obj2}?",
    "Identify the {color} object: {obj1} or {obj2}",
    "Which would be {color}: {obj1} or {obj2}?",
    "Between a {obj1} and a {obj2}, which is {color}?",
    "Is a {obj1} or a {obj2} {color}?",
    "Which is typically {color}, a {obj1} or a {obj2}?",
    "Point to the {color} one: {obj1} or {obj2}",
    "Select which is {color}: {obj1} or {obj2}",
    "Which object has {color} color: {obj1} or {obj2}?",
    "Of {obj1} and {obj2}, which one is {color}?",
    "Tell me which is {color}: {obj1} or {obj2}",
    "Which item is {color}, {obj1} or {obj2}?",
    "Name the {color} one: {obj1} or {obj2}",
    "Which appears {color}: {obj1} or {obj2}?",
    "Which looks {color}: {obj1} or {obj2}?",
    "Which do you call {color}: {obj1} or {obj2}?",
    "Indicate which is {color}: {obj1} or {obj2}",
    "Which would you say is {color}: {obj1} or {obj2}?",
    "Which naturally is {color}: {obj1} or {obj2}?",
    "Which is known to be {color}: {obj1} or {obj2}?",
]

# 1E: Color Mixing (expanded from 6 to 35+ templates)
TEMPLATES_1E = [
    # Basic questions
    'What color do you get when you mix "{color1}" and "{color2}"?',
    'If you mix "{color1}" and "{color2}", what color do you get?',
    'What is "{color1}" mixed with "{color2}"?',
    '"{color1}" plus "{color2}" makes what color?',
    'Combine "{color1}" and "{color2}", what color results?',
    'What happens when you mix "{color1}" with "{color2}"?',
    # Extended variations
    'What color results from mixing "{color1}" and "{color2}"?',
    'When "{color1}" and "{color2}" are combined, what color forms?',
    '"{color1}" and "{color2}" together make what color?',
    'What do you get from blending "{color1}" and "{color2}"?',
    'What is the result of mixing "{color1}" with "{color2}"?',
    'Mix "{color1}" and "{color2}", what color do you get?',
    'Blending "{color1}" and "{color2}" gives what color?',
    'If you combine "{color1}" with "{color2}", what results?',
    'What color appears when you blend "{color1}" and "{color2}"?',
    '"{color1}" combined with "{color2}" equals what color?',
    'What is produced by mixing "{color1}" and "{color2}"?',
    'Adding "{color1}" to "{color2}" creates what color?',
    '"{color1}" + "{color2}" = ?',
    'What does "{color1}" and "{color2}" make?',
    'Mixing "{color1}" with "{color2}" produces?',
    'What color is formed from "{color1}" and "{color2}"?',
    'The combination of "{color1}" and "{color2}" is?',
    'What new color comes from "{color1}" plus "{color2}"?',
    'When you put "{color1}" and "{color2}" together, what color?',
    '"{color1}" blended with "{color2}" creates?',
    'What shade results from "{color1}" and "{color2}"?',
    'Combining "{color1}" and "{color2}" gives?',
    'What is "{color1}" + "{color2}"?',
    'The mix of "{color1}" and "{color2}" is what color?',
    'If I blend "{color1}" and "{color2}", what do I get?',
    '"{color1}" mixed together with "{color2}" makes?',
    'What color do "{color1}" and "{color2}" create together?',
    'Name the color from mixing "{color1}" and "{color2}"',
    'Tell me what "{color1}" and "{color2}" make when mixed',
]

# 1F: Color Associations (expanded from 5 to 35+ templates)
TEMPLATES_1F = [
    # Basic questions
    "What color usually means {meaning}?",
    "What color represents {meaning}?",
    "Which color is associated with {meaning}?",
    "What color symbolizes {meaning}?",
    "{meaning} is typically shown in what color?",
    # Extended variations
    "What color do we use for {meaning}?",
    "What color is used to represent {meaning}?",
    "Which color stands for {meaning}?",
    "What color is commonly associated with {meaning}?",
    "{meaning} is what color?",
    "What color do people use for {meaning}?",
    "Which color means {meaning}?",
    "What color is linked to {meaning}?",
    "{meaning} is usually shown in which color?",
    "What color conveys {meaning}?",
    "What color is the symbol for {meaning}?",
    "Which color signifies {meaning}?",
    "What color denotes {meaning}?",
    "What color expresses {meaning}?",
    "{meaning} is indicated by what color?",
    "What color communicates {meaning}?",
    "Which color traditionally means {meaning}?",
    "What color would you use for {meaning}?",
    "What color is connected to {meaning}?",
    "{meaning} is represented by what color?",
    "What color is the sign for {meaning}?",
    "Which color relates to {meaning}?",
    "What color indicates {meaning}?",
    "What color is {meaning} usually shown in?",
    "What color do signs use for {meaning}?",
    "Traffic signs show {meaning} in what color?",
    "What color means {meaning} on signs?",
    "In symbols, what color represents {meaning}?",
    "What is the color for {meaning}?",
    "Tell me what color represents {meaning}",
]

# Statement 2: Shape Perception Templates

# 2A: Object Shape Identification (expanded from 10 to 40+ templates)
TEMPLATES_2A = [
    # Basic questions
    "What is the shape of a {obj}?",
    "What shape is a {obj}?",
    "Tell me the shape of a {obj}",
    "Identify the shape of a {obj}",
    "A {obj} has what shape?",
    "What shape does a {obj} have?",
    "Describe the shape of a {obj}",
    "Name the shape of a {obj}",
    "What geometric shape is a {obj}?",
    "What form does a {obj} have?",
    # Extended variations
    "What shape would you say a {obj} is?",
    "What is the form of a {obj}?",
    "A {obj} is what shape?",
    "What kind of shape is a {obj}?",
    "What type of shape does a {obj} have?",
    "What shape resembles a {obj}?",
    "If you trace a {obj}, what shape do you get?",
    "What is the outline shape of a {obj}?",
    "What is the typical shape of a {obj}?",
    "A {obj} has which geometric shape?",
    "What is a {obj} shaped like?",
    "What shape best describes a {obj}?",
    "What is the general shape of a {obj}?",
    "How would you describe the shape of a {obj}?",
    "Can you tell me the shape of a {obj}?",
    "What is the natural shape of a {obj}?",
    "What shape is associated with a {obj}?",
    "What geometric form is a {obj}?",
    "A {obj} resembles what shape?",
    "Looking at a {obj}, what shape is it?",
    "What shape does a {obj} look like?",
    "What shape appears when you see a {obj}?",
    "What would be the shape of a {obj}?",
    "In geometry, what shape is a {obj}?",
    "What polygon or shape is a {obj}?",
    "What is the basic shape of a {obj}?",
    "What shape characterizes a {obj}?",
    "What primary shape is a {obj}?",
    "The shape of a {obj} is?",
    "A {obj} has which shape?",
]

# 2A with adjectives (parametric expansion)
TEMPLATES_2A_ADJ = [
    "What is the shape of a {adj} {obj}?",
    "What shape is a {adj} {obj}?",
    "A {adj} {obj} has what shape?",
    "What shape does a {adj} {obj} have?",
    "Describe the shape of a {adj} {obj}",
    "What geometric shape is a {adj} {obj}?",
]

# 2A with context (parametric expansion)
TEMPLATES_2A_CONTEXT = [
    "What shape is a {obj} {context}?",
    "What is the shape of a {obj} found {context}?",
    "A {obj} {context} has what shape?",
    "If you see a {obj} {context}, what shape is it?",
]

# 2B: Reverse Shape Identification (expanded from 10 to 40+ templates)
TEMPLATES_2B = [
    # Basic questions
    "Name something that is {shape}",
    "Give me an object that is {shape}",
    "What is something that is {shape}?",
    "Tell me something {shape}",
    "Name a {shape} object",
    "What object is {shape}?",
    "Give an example of something {shape}",
    "What has the shape of a {shape}?",
    "Can you name something {shape}?",
    "List a {shape} object",
    # Extended variations
    "What thing is {shape}?",
    "Name an item that is {shape}",
    "What can you think of that is {shape}?",
    "What comes to mind that is {shape}?",
    "What is naturally {shape}?",
    "What do you know that is {shape}?",
    "Give me something that has a {shape} shape",
    "What item has a {shape} shape?",
    "Name a thing that is {shape}",
    "What's an example of something {shape}?",
    "Mention something that is {shape}",
    "Can you think of something {shape}?",
    "What would be {shape}?",
    "Give an instance of something {shape}",
    "What commonly appears {shape}?",
    "Point out something {shape}",
    "Identify something {shape}",
    "Provide an example of a {shape} thing",
    "What is known to be {shape}?",
    "What in nature is {shape}?",
    "What food is {shape}?",
    "What everyday object is {shape}?",
    "What item would you describe as {shape}?",
    "Something that is {shape} would be?",
    "What is shaped like a {shape}?",
    "What has a {shape} form?",
    "Name something with a {shape} shape",
    "What object has a {shape} outline?",
    "Give me a {shape}-shaped object",
    "What is typically {shape}?",
]

# 2B with context (parametric expansion)
TEMPLATES_2B_CONTEXT = [
    "Name something {shape} that you might find {context}",
    "What is {shape} {context}?",
    "Give me a {shape} object found {context}",
    "What {shape} thing would you see {context}?",
    "Name a {shape} item from {context}",
]

# 2C: Shape Verification (expanded from 5 to 30+ templates)
TEMPLATES_2C = [
    # Basic questions
    "Is a {obj} {shape}?",
    "Is the {obj} {shape}?",
    "Does a {obj} have a {shape} shape?",
    "Would you say a {obj} is {shape}?",
    "Is {shape} the shape of a {obj}?",
    # Extended variations
    "Is it true that a {obj} is {shape}?",
    "Can a {obj} be described as {shape}?",
    "Is a {obj} typically {shape}?",
    "Are {obj_plural} {shape}?",
    "Do {obj_plural} have a {shape} shape?",
    "Is {shape} a common shape for a {obj}?",
    "Would a {obj} be considered {shape}?",
    "Is a {obj} known to be {shape}?",
    "Would you describe a {obj} as {shape}?",
    "Is it correct that a {obj} is {shape}?",
    "True or false: a {obj} is {shape}",
    "Is a {obj} usually {shape}?",
    "Is the shape of a {obj} {shape}?",
    "Could you say a {obj} is {shape}?",
    "Is a {obj} generally {shape}?",
    "Would a {obj} appear {shape}?",
    "Is a {obj} naturally {shape}?",
    "Is {shape} what shape a {obj} is?",
    "A {obj} is {shape}, right?",
    "Can we say a {obj} is {shape}?",
    "Is it accurate that a {obj} is {shape}?",
    "Would it be correct to call a {obj} {shape}?",
    "Is the {obj} {shape}?",
    "Does a {obj} come in a {shape} shape?",
    "Is a {obj} shaped like a {shape}?",
]

# 2D: Shape Multiple Choice (expanded from 5 to 30+ templates)
TEMPLATES_2D = [
    # Basic questions
    "Which is {shape}, a {obj1} or a {obj2}?",
    "Which one is {shape}: {obj1} or {obj2}?",
    "Pick the {shape} object: {obj1} or {obj2}",
    "Between {obj1} and {obj2}, which is {shape}?",
    "Is {obj1} or {obj2} {shape}?",
    # Extended variations
    "Which of these is {shape}: {obj1} or {obj2}?",
    "Select the {shape} item: {obj1} or {obj2}?",
    "Which has a {shape} shape, {obj1} or {obj2}?",
    "Out of {obj1} and {obj2}, which is {shape}?",
    "Choose the {shape} one: {obj1} or {obj2}",
    "What is {shape}, {obj1} or {obj2}?",
    "Identify the {shape} object: {obj1} or {obj2}",
    "Which would be {shape}: {obj1} or {obj2}?",
    "Between a {obj1} and a {obj2}, which is {shape}?",
    "Is a {obj1} or a {obj2} {shape}?",
    "Which is typically {shape}, a {obj1} or a {obj2}?",
    "Point to the {shape} one: {obj1} or {obj2}",
    "Select which is {shape}: {obj1} or {obj2}",
    "Which object has a {shape} shape: {obj1} or {obj2}?",
    "Of {obj1} and {obj2}, which one is {shape}?",
    "Tell me which is {shape}: {obj1} or {obj2}",
    "Which item is {shape}, {obj1} or {obj2}?",
    "Name the {shape} one: {obj1} or {obj2}",
    "Which appears {shape}: {obj1} or {obj2}?",
    "Which looks {shape}: {obj1} or {obj2}?",
    "Which is shaped like a {shape}: {obj1} or {obj2}?",
    "Indicate which is {shape}: {obj1} or {obj2}",
    "Which would you say is {shape}: {obj1} or {obj2}?",
    "Which has the {shape} form: {obj1} or {obj2}?",
    "Which is known to be {shape}: {obj1} or {obj2}?",
]

# 2E: 2D vs 3D Distinction (expanded from 5 to 30+ templates)
TEMPLATES_2E = [
    # Basic questions
    "Is a {shape} a 2D shape or 3D shape?",
    "Is {shape} two-dimensional or three-dimensional?",
    "Is a {obj} flat or solid?",
    "Would you classify {shape} as 2D or 3D?",
    "Is {obj} a flat shape or a solid shape?",
    # Extended variations
    "Is a {shape} 2D or 3D?",
    "Is {shape} flat or three-dimensional?",
    "Does a {shape} have two dimensions or three?",
    "Is {shape} considered 2D or 3D?",
    "Would a {shape} be 2D or 3D?",
    "Is a {shape} a plane figure or a solid figure?",
    "Is {shape} a plane shape or a solid shape?",
    "Does {shape} have depth or is it flat?",
    "Is a {shape} flat like paper or solid like a box?",
    "Would you say {shape} is 2D or 3D?",
    "Is a {shape} two-dimensional or has it got depth?",
    "A {shape} is 2D or 3D?",
    "Is the {shape} a flat shape or solid shape?",
    "Does a {shape} exist in 2D or 3D?",
    "Is a {shape} planar or spatial?",
    "Would {shape} be considered flat or solid?",
    "Is {shape} drawn in 2D or modeled in 3D?",
    "Is a {shape} a 2-dimensional or 3-dimensional shape?",
    "Does {shape} have three dimensions or two?",
    "Is a {shape} like a drawing or like a model?",
    "Is {shape} a polygon or a polyhedron?",
    "Is the shape {shape} 2D or 3D?",
    "Does a {shape} occupy space or is it flat?",
    "Is {shape} in the category of 2D or 3D shapes?",
    "Would {shape} appear on paper as 2D or need 3D space?",
]

# 2F: 2D-3D Relationship (expanded)
TEMPLATES_2F_3D_TO_2D = [
    "What is the 2D outline of a {shape3d}?",
    "What 2D shape is the face of a {shape3d}?",
    "If you trace around a {shape3d}, what 2D shape do you get?",
    "What flat shape is the base of a {shape3d}?",
    "What 2D shape forms the face of a {shape3d}?",
    "Looking at a {shape3d} from one side, what 2D shape do you see?",
    "What is the cross-section of a {shape3d}?",
    "What 2D shape appears on the surface of a {shape3d}?",
    "The face of a {shape3d} is which 2D shape?",
    "What flat shape can be found on a {shape3d}?",
    "If you slice a {shape3d}, what shape might you see?",
    "What 2D shape makes up the sides of a {shape3d}?",
    "The base of a {shape3d} is what shape?",
    "What flat geometric shape is part of a {shape3d}?",
    "If you unfold a {shape3d}, what 2D shapes appear?",
]

TEMPLATES_2F_2D_TO_3D = [
    "What 3D shape has {shape2d} faces?",
    "What 3D shape is made of {shape2d}s?",
    "What solid shape has {shape2d} as a face?",
    "Which 3D shape uses {shape2d}s?",
    "What 3D object has faces that are {shape2d}s?",
    "Name a 3D shape with {shape2d} faces",
    "What solid has {shape2d}s as its sides?",
    "A 3D shape made of {shape2d}s is called?",
    "What polyhedron has {shape2d} faces?",
    "Which 3D object is constructed from {shape2d}s?",
    "What solid figure has {shape2d} as its base?",
    "Name a solid shape that uses {shape2d}s",
    "What 3D shape can be built from {shape2d}s?",
    "Which solid has faces shaped like {shape2d}s?",
    "A 3D object with {shape2d} sides is what?",
]

# Statement 3: Geometric Concepts Templates

# 3A: Shape by Sides (expanded from 8 to 30+ templates)
TEMPLATES_3A = [
    # Basic questions
    "Which shape has {n} sides?",
    "What shape has {n} sides?",
    "Name the shape with {n} sides",
    "A shape with {n} sides is called?",
    "What do you call a shape with {n} sides?",
    "Which polygon has {n} sides?",
    "Identify the shape with {n} sides",
    "{n}-sided shape is called?",
    # Extended variations
    "What is the name of a {n}-sided shape?",
    "What polygon has exactly {n} sides?",
    "A {n}-sided polygon is known as?",
    "What is a shape with {n} sides called?",
    "Tell me the name of a shape with {n} sides",
    "What figure has {n} sides?",
    "A polygon with {n} sides is a?",
    "What geometric shape has {n} sides?",
    "Which figure has {n} sides?",
    "The shape with {n} sides is called?",
    "Name a polygon with {n} sides",
    "What do we call a polygon with {n} sides?",
    "A figure with {n} sides is known as?",
    "What is the {n}-sided polygon called?",
    "Identify the polygon with {n} sides",
    "What shape has exactly {n} sides?",
    "A shape that has {n} sides is a?",
    "Can you name the {n}-sided shape?",
    "The {n}-sided figure is called?",
    "What is a {n}-gon?",
    "A regular shape with {n} sides is?",
]

# 3B: Sides of Shape (expanded from 5 to 30+ templates)
TEMPLATES_3B = [
    # Basic questions
    "How many sides does a {shape} have?",
    "Count the sides of a {shape}",
    "A {shape} has how many sides?",
    "What is the number of sides in a {shape}?",
    "Tell me the side count of a {shape}",
    # Extended variations
    "How many sides are there in a {shape}?",
    "What is the side count of a {shape}?",
    "A {shape} contains how many sides?",
    "Count how many sides a {shape} has",
    "The number of sides in a {shape} is?",
    "How many edges does a {shape} have?",
    "A {shape} has how many straight edges?",
    "What is the total number of sides in a {shape}?",
    "How many line segments form a {shape}?",
    "In a {shape}, how many sides are there?",
    "Tell me the number of sides of a {shape}",
    "A {shape} is made of how many sides?",
    "What is the count of sides in a {shape}?",
    "How many sides make up a {shape}?",
    "The side count of a {shape} is?",
    "A {shape} consists of how many sides?",
    "How many sides does the shape {shape} have?",
    "For a {shape}, how many sides are there?",
    "What number of sides does a {shape} have?",
    "Name the number of sides in a {shape}",
    "A {shape} is bounded by how many sides?",
    "How many straight sides does a {shape} have?",
    "Count the number of sides on a {shape}",
    "State the number of sides of a {shape}",
    "How many sides can you count on a {shape}?",
]

# 3C: Corners/Vertices (2D and 3D) (expanded from 6 to 30+ templates)
TEMPLATES_3C = [
    # Basic questions
    "How many corners does a {shape} have?",
    "How many vertices does a {shape} have?",
    "Count the corners of a {shape}",
    "A {shape} has how many corners?",
    "What is the number of vertices in a {shape}?",
    "Tell me the corner count of a {shape}",
    # Extended variations
    "How many corners are there in a {shape}?",
    "What is the corner count of a {shape}?",
    "A {shape} contains how many corners?",
    "Count how many corners a {shape} has",
    "The number of corners in a {shape} is?",
    "How many points does a {shape} have?",
    "A {shape} has how many vertex points?",
    "What is the total number of corners in a {shape}?",
    "How many corner points form a {shape}?",
    "In a {shape}, how many corners are there?",
    "Tell me the number of corners of a {shape}",
    "A {shape} is made of how many corners?",
    "What is the count of corners in a {shape}?",
    "How many corners make up a {shape}?",
    "The corner count of a {shape} is?",
    "A {shape} consists of how many corners?",
    "How many vertices does the shape {shape} have?",
    "For a {shape}, how many corners are there?",
    "What number of corners does a {shape} have?",
    "Name the number of corners in a {shape}",
    "How many vertex points does a {shape} have?",
    "Count the number of vertices on a {shape}",
    "State the number of corners of a {shape}",
    "How many corners can you count on a {shape}?",
]

# 3D: Angles Count (expanded from 4 to 25+ templates)
TEMPLATES_3D_ANGLES = [
    # Basic questions
    "How many angles does a {shape} have?",
    "Count the angles in a {shape}",
    "A {shape} has how many angles?",
    "What is the number of angles in a {shape}?",
    # Extended variations
    "How many angles are there in a {shape}?",
    "What is the angle count of a {shape}?",
    "A {shape} contains how many angles?",
    "Count how many angles a {shape} has",
    "The number of angles in a {shape} is?",
    "A {shape} has how many interior angles?",
    "What is the total number of angles in a {shape}?",
    "In a {shape}, how many angles are there?",
    "Tell me the number of angles of a {shape}",
    "What is the count of angles in a {shape}?",
    "How many angles make up a {shape}?",
    "The angle count of a {shape} is?",
    "A {shape} consists of how many angles?",
    "How many angles does the shape {shape} have?",
    "For a {shape}, how many angles are there?",
    "What number of angles does a {shape} have?",
    "Name the number of angles in a {shape}",
    "Count the number of angles on a {shape}",
    "State the number of angles of a {shape}",
    "How many angles can you count on a {shape}?",
    "A {shape} is bounded by how many angles?",
]

# 3E: Angle Types (expanded from 12 to 50+ templates)
TEMPLATES_3E_NAME = [
    # Acute angle questions
    "What is an angle less than 90 degrees called?",
    "What do we call an angle under 90 degrees?",
    "An angle smaller than 90 degrees is called?",
    "Name the type of angle that is less than 90 degrees",
    "What is the name for an angle less than 90 degrees?",
    "An angle that measures less than 90 degrees is a?",
    "What type of angle is less than 90 degrees?",
    "An angle under 90 degrees is known as?",
    "What is an angle below 90 degrees called?",
    "An angle measuring less than 90 degrees is called?",
    # Right angle questions
    "What is an angle equal to 90 degrees called?",
    "What do we call an angle of exactly 90 degrees?",
    "An angle that measures 90 degrees is called?",
    "Name the type of angle that equals 90 degrees",
    "What is the name for a 90 degree angle?",
    "An angle that is exactly 90 degrees is a?",
    "What type of angle measures exactly 90 degrees?",
    "A 90 degree angle is known as?",
    "What is a 90 degree angle called?",
    # Obtuse angle questions
    "What is an angle greater than 90 degrees but less than 180 degrees called?",
    "What do we call an angle between 90 and 180 degrees?",
    "An angle larger than 90 but smaller than 180 degrees is called?",
    "Name the type of angle that is between 90 and 180 degrees",
    "What is the name for an angle more than 90 but less than 180?",
    "An angle measuring more than 90 but less than 180 degrees is a?",
    "What type of angle is greater than 90 but less than 180?",
    "An angle over 90 but under 180 degrees is known as?",
    # Straight angle questions
    "What is an angle equal to 180 degrees called?",
    "What do we call an angle of exactly 180 degrees?",
    "An angle that measures 180 degrees is called?",
    "Name the type of angle that equals 180 degrees",
    "What is the name for a 180 degree angle?",
    "An angle that is exactly 180 degrees is a?",
    "What type of angle measures exactly 180 degrees?",
    "A 180 degree angle is known as?",
    # Reflex angle questions
    "What is an angle greater than 180 degrees called?",
    "What do we call an angle over 180 degrees?",
    "An angle larger than 180 degrees is called?",
    "Name the type of angle that is more than 180 degrees",
    "What is the name for an angle greater than 180?",
    "An angle measuring more than 180 degrees is a?",
    "What type of angle is greater than 180 degrees?",
    "An angle over 180 degrees is known as?",
    "What is an angle between 180 and 360 degrees called?",
]

TEMPLATES_3E_DEF = [
    # Acute angle definitions
    "What is an acute angle?",
    "Define an acute angle",
    "What makes an angle acute?",
    "How do you describe an acute angle?",
    "What are the characteristics of an acute angle?",
    "An acute angle is defined as?",
    # Right angle definitions
    "How many degrees is a right angle?",
    "What is the measure of a right angle?",
    "A right angle measures how many degrees?",
    "What is the degree measure of a right angle?",
    "Define a right angle",
    "What is a right angle?",
    # Obtuse angle definitions
    "What is an obtuse angle?",
    "Define an obtuse angle",
    "What makes an angle obtuse?",
    "How do you describe an obtuse angle?",
    "What are the characteristics of an obtuse angle?",
    "An obtuse angle is defined as?",
    # Straight angle definitions
    "What is a straight angle?",
    "Define a straight angle",
    "What makes an angle straight?",
    "How do you describe a straight angle?",
    "A straight angle measures how many degrees?",
    "What is the measure of a straight angle?",
    # Reflex angle definitions
    "What is a reflex angle?",
    "Define a reflex angle",
    "What makes an angle a reflex angle?",
    "How do you describe a reflex angle?",
]

# 3F: Shape Properties (expanded from 4 to 25+ templates)
TEMPLATES_3F = [
    # Basic questions
    "Which shape has {property}?",
    "What shape has {property}?",
    "Name the shape with {property}",
    "A shape with {property} is called?",
    # Extended variations
    "What is the shape that has {property}?",
    "Which polygon has {property}?",
    "Identify the shape with {property}",
    "What figure has {property}?",
    "Name a shape that has {property}",
    "What geometric shape has {property}?",
    "Which shape is known for having {property}?",
    "What shape features {property}?",
    "A shape characterized by {property} is?",
    "What shape is defined by having {property}?",
    "Which shape includes {property}?",
    "Tell me a shape with {property}",
    "What shape possesses {property}?",
    "Which figure features {property}?",
    "Identify a shape having {property}",
    "What shape can be described as having {property}?",
    "Name the geometric shape with {property}",
    "What shape is notable for {property}?",
    "Which shape contains {property}?",
    "A figure with {property} is what shape?",
]

# 3G: 3D Faces/Edges/Vertices (expanded from 6 to 35+ templates)
TEMPLATES_3G = [
    # Faces questions
    "How many faces does a {shape3d} have?",
    "Count the faces of a {shape3d}",
    "A {shape3d} has how many faces?",
    "What is the number of faces in a {shape3d}?",
    "Tell me the face count of a {shape3d}",
    "How many flat surfaces does a {shape3d} have?",
    "How many sides does a {shape3d} have?",
    "What is the face count of a {shape3d}?",
    "A {shape3d} contains how many faces?",
    "In a {shape3d}, how many faces are there?",
    "The number of faces on a {shape3d} is?",
    "How many flat faces make up a {shape3d}?",
    # Edges questions
    "How many edges does a {shape3d} have?",
    "Count the edges of a {shape3d}",
    "A {shape3d} has how many edges?",
    "What is the number of edges in a {shape3d}?",
    "Tell me the edge count of a {shape3d}",
    "How many line edges does a {shape3d} have?",
    "What is the edge count of a {shape3d}?",
    "A {shape3d} contains how many edges?",
    "In a {shape3d}, how many edges are there?",
    "The number of edges on a {shape3d} is?",
    "How many edges make up a {shape3d}?",
    # Vertices questions
    "How many vertices does a {shape3d} have?",
    "Count the vertices of a {shape3d}",
    "A {shape3d} has how many vertices?",
    "What is the number of vertices in a {shape3d}?",
    "Tell me the vertex count of a {shape3d}",
    "How many corner points does a {shape3d} have?",
    "What is the vertex count of a {shape3d}?",
    "A {shape3d} contains how many vertices?",
    "In a {shape3d}, how many vertices are there?",
    "The number of vertices on a {shape3d} is?",
    "How many corners does a {shape3d} have?",
    "How many points does a {shape3d} have?",
]

# 3H: Symmetry (expanded from 4 to 25+ templates)
TEMPLATES_3H = [
    # Basic questions
    "How many lines of symmetry does a {shape} have?",
    "Count the lines of symmetry in a {shape}",
    "A {shape} has how many axes of symmetry?",
    "What is the number of symmetry lines in a {shape}?",
    # Extended variations
    "How many axes of symmetry does a {shape} have?",
    "Tell me the number of lines of symmetry in a {shape}",
    "A {shape} contains how many lines of symmetry?",
    "What is the symmetry count of a {shape}?",
    "How many symmetry axes are in a {shape}?",
    "Count how many lines of symmetry a {shape} has",
    "The number of symmetry lines in a {shape} is?",
    "For a {shape}, how many lines of symmetry exist?",
    "How many mirror lines does a {shape} have?",
    "A {shape} is symmetric along how many lines?",
    "What is the line count for symmetry in a {shape}?",
    "How many times can you fold a {shape} symmetrically?",
    "Name the number of symmetry lines in a {shape}",
    "A {shape} has how many reflection lines?",
    "Count the symmetry axes of a {shape}",
    "How many ways can a {shape} be folded symmetrically?",
    "What is the axis of symmetry count for a {shape}?",
    "In a {shape}, how many lines of symmetry are there?",
    "The symmetry line count of a {shape} is?",
    "A {shape} possesses how many symmetry lines?",
    "How many lines divide a {shape} symmetrically?",
]

# 3I: Shape Comparisons (expanded from 4 to 30+ templates)
TEMPLATES_3I = [
    # Sides comparisons
    'Which has more sides, a "{shape1}" or a "{shape2}"?',
    'Which has fewer sides, a "{shape1}" or a "{shape2}"?',
    'Between a "{shape1}" and a "{shape2}", which has more sides?',
    'Compare a "{shape1}" and a "{shape2}", which has more sides?',
    'Which shape has more sides: "{shape1}" or "{shape2}"?',
    'Which shape has fewer sides: "{shape1}" or "{shape2}"?',
    'Does a "{shape1}" or a "{shape2}" have more sides?',
    'Which one has more sides, "{shape1}" or "{shape2}"?',
    'Of "{shape1}" and "{shape2}", which has more sides?',
    'Is it "{shape1}" or "{shape2}" that has more sides?',
    # Corners comparisons
    'Which has more corners, a "{shape1}" or a "{shape2}"?',
    'Which has fewer corners, a "{shape1}" or a "{shape2}"?',
    'Between a "{shape1}" and a "{shape2}", which has more corners?',
    'Compare a "{shape1}" and a "{shape2}", which has more corners?',
    'Which shape has more corners: "{shape1}" or "{shape2}"?',
    'Which shape has fewer corners: "{shape1}" or "{shape2}"?',
    'Does a "{shape1}" or a "{shape2}" have more corners?',
    'Which one has more corners, "{shape1}" or "{shape2}"?',
    'Of "{shape1}" and "{shape2}", which has more corners?',
    # Angles comparisons
    'Which has more angles, a "{shape1}" or a "{shape2}"?',
    'Which has fewer angles, a "{shape1}" or a "{shape2}"?',
    'Between a "{shape1}" and a "{shape2}", which has more angles?',
    'Compare a "{shape1}" and a "{shape2}", which has more angles?',
    'Which shape has more angles: "{shape1}" or "{shape2}"?',
    'Which shape has fewer angles: "{shape1}" or "{shape2}"?',
    'Does a "{shape1}" or a "{shape2}" have more angles?',
    'Which one has more angles, "{shape1}" or "{shape2}"?',
    'Of "{shape1}" and "{shape2}", which has more angles?',
    'Is it "{shape1}" or "{shape2}" that has more angles?',
    # General comparisons
    'Which is larger in terms of sides, "{shape1}" or "{shape2}"?',
    'Which polygon has more sides: "{shape1}" or "{shape2}"?',
]

# 3J: Circle Properties (expanded from 6 to 35+ templates)
TEMPLATES_3J = [
    # Circumference questions
    "What is the distance around a circle called?",
    "What is the perimeter of a circle called?",
    "What do we call the outline of a circle?",
    "The distance around a circle is known as?",
    "What is the name for the boundary of a circle?",
    "What term describes the distance around a circle?",
    "The perimeter of a circle is called?",
    "What is another name for the perimeter of a circle?",
    # Diameter questions
    "What is the distance across a circle through the center called?",
    "What is a line from one side of a circle to the other through the center?",
    "What do we call a line crossing the center of a circle?",
    "The distance across a circle through its center is?",
    "What is the longest distance across a circle called?",
    "A line segment through the center of a circle is called?",
    "What is the name for a line crossing the center of a circle?",
    "The width of a circle through the center is known as?",
    # Radius questions
    "What is half of the diameter called?",
    "What do we call the distance from center to edge of a circle?",
    "What is the distance from the center of a circle to its edge?",
    "Half of a circle's diameter is called?",
    "The distance from a circle's center to its boundary is?",
    "What is the name for half the diameter of a circle?",
    "A line from center to edge of a circle is called?",
    "What do we call the line from center to the circumference?",
    # General circle questions
    "What is the center point of a circle called?",
    "What do we call any point inside a circle equidistant from the edge?",
    "What is Pi used to calculate for a circle?",
    "What Greek letter is used in circle calculations?",
    "What connects the center of a circle to its edge?",
    "What is the relationship between diameter and radius?",
    "How is circumference related to diameter?",
    "What is twice the radius of a circle?",
    "What is half the width of a circle called?",
    "The line from center to circumference is the?",
]

# 3K: Basic Perimeter (expanded from 5 to 30+ templates)
TEMPLATES_3K_SQUARE = [
    "If each side of a square is {n}, what is the perimeter?",
    "A square has sides of length {n}. What is its perimeter?",
    "Find the perimeter of a square with side {n}",
    "Calculate the perimeter of a square with side length {n}",
    "What is the perimeter of a square with {n} unit sides?",
    "A square with side {n} has what perimeter?",
    "Compute the perimeter of a square where each side is {n}",
    "The perimeter of a square with sides of {n} is?",
    "If a square has sides of {n}, find its perimeter",
    "What's the total distance around a square with side {n}?",
    "For a square with sides measuring {n}, what is the perimeter?",
    "A {n}-unit square has what perimeter?",
    "Determine the perimeter of a square with side {n}",
    "What is the perimeter if a square's side is {n}?",
    "Calculate: perimeter of a square with side = {n}",
]

TEMPLATES_3K_RECT = [
    "A rectangle has length {l} and width {w}, what is its perimeter?",
    "Calculate the perimeter of a rectangle {l} by {w}",
    "Find the perimeter of a {l} × {w} rectangle",
    "What is the perimeter of a rectangle with length {l} and width {w}?",
    "A {l} by {w} rectangle has what perimeter?",
    "Compute the perimeter of a rectangle measuring {l} by {w}",
    "The perimeter of a rectangle {l} long and {w} wide is?",
    "If a rectangle is {l} by {w}, what is its perimeter?",
    "What's the total distance around a {l} × {w} rectangle?",
    "For a rectangle with dimensions {l} and {w}, what is the perimeter?",
    "Determine the perimeter of a rectangle {l} units by {w} units",
    "What is the perimeter of a rectangle measuring {l} by {w}?",
    "Calculate: perimeter of rectangle with length {l}, width {w}",
    "A rectangle of {l} × {w} has perimeter equal to?",
    "Find: perimeter of a {l} by {w} rectangle",
]

# 3L: Basic Area (expanded from 5 to 30+ templates)
TEMPLATES_3L_SQUARE = [
    "What is the area of a square with side {n}?",
    "Find the area of a square with sides of {n}",
    "Calculate the area of a square with side length {n}",
    "A square has sides of length {n}. What is its area?",
    "What is the area of a {n}-unit square?",
    "A square with side {n} has what area?",
    "Compute the area of a square where each side is {n}",
    "The area of a square with sides of {n} is?",
    "If a square has sides of {n}, find its area",
    "What's the area covered by a square with side {n}?",
    "For a square with sides measuring {n}, what is the area?",
    "A {n} × {n} square has what area?",
    "Determine the area of a square with side {n}",
    "What is the area if a square's side is {n}?",
    "Calculate: area of a square with side = {n}",
]

TEMPLATES_3L_RECT = [
    "A rectangle is {l} long and {w} wide, what is its area?",
    "Calculate the area of a rectangle {l} by {w}",
    "What is the area of a {l} × {w} rectangle?",
    "Find the area of a rectangle with length {l} and width {w}",
    "A {l} by {w} rectangle has what area?",
    "Compute the area of a rectangle measuring {l} by {w}",
    "The area of a rectangle {l} long and {w} wide is?",
    "If a rectangle is {l} by {w}, what is its area?",
    "What's the area covered by a {l} × {w} rectangle?",
    "For a rectangle with dimensions {l} and {w}, what is the area?",
    "Determine the area of a rectangle {l} units by {w} units",
    "What is the area of a rectangle measuring {l} by {w}?",
    "Calculate: area of rectangle with length {l}, width {w}",
    "A rectangle of {l} × {w} has area equal to?",
    "Find: area of a {l} by {w} rectangle",
]

# Statement 4: Pattern Recognition Templates

# 4A-4K: Pattern completion (expanded from 6 to 40+ templates)
TEMPLATES_PATTERN = [
    # Basic questions
    'What comes next: "{pattern}"?',
    'What should come next: "{pattern}"?',
]

# Additional pattern question styles with different phrasings
TEMPLATES_PATTERN_ALT = [
    'Look at this pattern: "{pattern}", what comes next?',
    'Given the sequence "{pattern}", what follows?',
    'In the pattern "{pattern}", what is the next item?',
    'Observe this pattern: "{pattern}", what comes next?',
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_color_for_object(obj: str) -> str:
    """Get the color for a given object."""
    for color, objects in COLOR_OBJECTS.items():
        if obj in objects:
            return color
    return None


def get_shape_for_object(obj: str) -> str:
    """Get the shape for a given object."""
    for shape, objects in SHAPE_OBJECTS_2D.items():
        if obj in objects:
            return shape
    for shape, objects in SHAPE_OBJECTS_3D.items():
        if obj in objects:
            return shape
    return None


def format_pattern(items: List) -> str:
    """Format a pattern list as a simple comma-separated string."""
    return ", ".join(str(item) for item in items)


# ============================================================================
# STATEMENT 1: COLOR PERCEPTION GENERATORS
# ============================================================================

# Uncountable nouns that don't take articles
UNCOUNTABLE_NOUNS = {
    "wool",
    "butter",
    "milk",
    "water",
    "cheese",
    "bread",
    "rice",
    "sugar",
    "salt",
    "coffee",
    "tea",
    "juice",
    "honey",
    "dust",
    "dirt",
    "soil",
    "mud",
    "grass",
    "hair",
    "money",
    "paper",
    "wood",
    "glass",
    "metal",
    "plastic",
    "cotton",
    "silk",
    "leather",
    "stone",
    "rock",
    "concrete",
    "steel",
    "gold",
    "silver",
    "ink",
    "paint",
    "blood",
    "water",
    "air",
    "fire",
    "smoke",
    "ash",
}


# Helper function to check if object is uncountable
def is_uncountable(obj: str) -> bool:
    """Check if object is an uncountable noun."""
    obj_lower = obj.lower().strip()
    return obj_lower in UNCOUNTABLE_NOUNS


# Helper function to get correct article (a/an) for objects
def get_article(obj: str) -> str:
    """Return 'a' or 'an' based on the first letter/sound of the object."""
    # Uncountable nouns don't take articles
    if is_uncountable(obj):
        return ""

    obj_lower = obj.lower().strip()
    if not obj_lower:
        return "a"

    # Words starting with vowels (a, e, i, o, u) use "an"
    # Also handle silent 'h' cases
    first_char = obj_lower[0]
    if first_char in "aeiou":
        return "an"
    # Special cases: silent 'h' words
    if obj_lower.startswith(("hour", "honor", "honest", "heir")):
        return "an"
    return "a"


# Helper function to pluralize objects for simple worksheet prompts
def pluralize_simple(obj: str) -> str:
    """
    Naive English pluralization for short worksheet prompts.
    Works on the last character(s) of the full phrase, which is good enough for
    phrases like 'fire truck' -> 'fire trucks'.
    """
    obj_lower = obj.lower().strip()
    if not obj_lower:
        return obj
    # If it already looks plural, keep it (avoid 'mangos' -> 'mangoss')
    if obj_lower.endswith("s"):
        return obj
    if obj_lower.endswith(("ch", "sh", "x", "z")):
        return obj + "es"
    if obj_lower.endswith("y") and len(obj_lower) > 1 and obj_lower[-2] not in "aeiou":
        return obj[:-1] + "ies"
    return obj + "s"


# Helper function to format object with correct article
def format_object(obj: str, template: str) -> str:
    """Format template with object, handling articles correctly."""
    # Check if template uses {obj} or {article_obj}
    if "{article_obj}" in template:
        article = get_article(obj)
        return template.replace("{article_obj}", f"{article} {obj}")
    elif "{obj_plural}" in template:
        return template.replace("{obj_plural}", pluralize_simple(obj))
    elif "{obj}" in template:
        # For templates that already have article, use as-is
        return template.replace("{obj}", obj)
    else:
        return template


def generate_s1a_object_color_id(num_samples: int = 35000) -> Dict[str, str]:
    """1A: Object Color Identification (35,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    while len(samples) < num_samples and attempt < max_attempts:
        color = random.choice(ALL_COLORS)
        obj = random.choice(COLOR_OBJECTS[color])

        # Use ONLY base templates (no adjectives, no contexts - creates unnatural phrasing)
        template = random.choice(TEMPLATES_1A)

        # Handle articles correctly
        if "{article_obj}" in template:
            # Template uses {article_obj} placeholder
            if is_uncountable(obj):
                # No article for uncountable nouns
                query = template.replace("{article_obj}", obj)
            else:
                article = get_article(obj)
                query = template.replace("{article_obj}", f"{article} {obj}")
        elif "{obj}" in template:
            # Template uses {obj} - need to handle articles
            if is_uncountable(obj):
                # Remove articles for uncountable nouns (handle both "a" and "an")
                query = template.replace(" an {obj}", f" {obj}").replace(
                    " a {obj}", f" {obj}"
                )
                query = query.replace("An {obj}", obj.capitalize()).replace(
                    "A {obj}", obj.capitalize()
                )
                query = query.replace("{obj}", obj)  # Handle any remaining {obj}
            elif (
                " a {obj}" in template
                or "is a {obj}" in template
                or "does a {obj}" in template
            ):
                # Template has "a" - check if we need "an" instead
                article = get_article(obj)
                if article == "an":
                    query = template.replace("a {obj}", f"an {obj}")
                    query = query.replace("{obj}", obj)
                else:
                    query = template.replace("{obj}", obj)
            elif (
                " an {obj}" in template
                or "is an {obj}" in template
                or "does an {obj}" in template
            ):
                # Template has "an" - check if we need "a" instead
                article = get_article(obj)
                if article == "a":
                    query = template.replace("an {obj}", f"a {obj}")
                    query = query.replace("{obj}", obj)
                else:
                    query = template.replace("{obj}", obj)
            elif "are {obj}" in template or "are {obj}?" in template:
                # Plural form
                plural_obj = pluralize_simple(obj)
                query = template.replace("{obj}", plural_obj)
            elif "the {obj}" in template:
                # Definite article - use as-is
                query = template.replace("{obj}", obj)
            else:
                # Default - just replace (no article in template)
                query = template.replace("{obj}", obj)
        else:
            query = template

        answer = color

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s1b_reverse_color_id(num_samples: int = 15000) -> Dict[str, str]:
    """1B: Reverse Color Identification (15,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    while len(samples) < num_samples and attempt < max_attempts:
        color = random.choice(ALL_COLORS)

        # Use only base templates (no context - creates unnatural phrasing)
        template = random.choice(TEMPLATES_1B)
        query = template.format(color=color)

        # Pick a random object of this color
        answer = random.choice(COLOR_OBJECTS[color])

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s1c_color_verification(num_samples: int = 8000) -> Dict[str, str]:
    """1C: Color Verification (8,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    # 70% true, 30% false
    true_count = int(num_samples * 0.7)
    current_true = 0

    while len(samples) < num_samples and attempt < max_attempts:
        is_true = current_true < true_count

        if is_true:
            # Pick a correct color-object pair
            color = random.choice(ALL_COLORS)
            obj = random.choice(COLOR_OBJECTS[color])
        else:
            # Pick an incorrect pair
            color = random.choice(ALL_COLORS)
            # Pick an object from a different color
            wrong_color = random.choice([c for c in ALL_COLORS if c != color])
            obj = random.choice(COLOR_OBJECTS[wrong_color])

        # Randomly choose template type (75% base, 25% with adjective)
        template_type = random.random()

        if template_type < 0.75:
            # Base template
            template = random.choice(TEMPLATES_1C)
            query = template.format(
                obj=obj, obj_plural=pluralize_simple(obj), color=color
            )

            # Fix article usage for uncountable nouns and a/an for vowel-starting objects.
            if is_uncountable(obj):
                query = query.replace(f" an {obj}", f" {obj}").replace(
                    f" a {obj}", f" {obj}"
                )
                query = query.replace(f"An {obj}", obj.capitalize()).replace(
                    f"A {obj}", obj.capitalize()
                )
            else:
                article = get_article(obj)
                if article == "an":
                    query = query.replace(f"a {obj}", f"an {obj}")
                else:
                    query = query.replace(f"an {obj}", f"a {obj}")
        else:
            # Template with adjective
            adj = random.choice(ADJECTIVES)
            template = random.choice(TEMPLATES_1C_ADJ)
            # Avoid adjective templates for uncountable nouns (e.g. "a big water")
            if is_uncountable(obj):
                attempt += 1
                continue
            query = template.format(adj=adj, obj=obj, color=color)

        answer = "yes" if is_true else "no"

        if query not in samples:
            samples[query] = answer
            if is_true:
                current_true += 1

        attempt += 1

    return samples


def generate_s1d_color_multiple_choice(num_samples: int = 5000) -> Dict[str, str]:
    """1D: Color Multiple Choice (5,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    while len(samples) < num_samples and attempt < max_attempts:
        color = random.choice(ALL_COLORS)
        # Pick one object with this color
        obj1 = random.choice(COLOR_OBJECTS[color])
        # Pick another object with a different color
        other_color = random.choice([c for c in ALL_COLORS if c != color])
        obj2 = random.choice(COLOR_OBJECTS[other_color])

        # Randomize order
        if random.random() < 0.5:
            obj1, obj2 = obj2, obj1

        template = random.choice(TEMPLATES_1D)
        query = template.format(color=color, obj1=obj1, obj2=obj2)

        # Answer is the object that has the target color
        answer = obj1 if get_color_for_object(obj1) == color else obj2

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s1e_color_mixing(num_samples: int = 4000) -> Dict[str, str]:
    """1E: Color Mixing (4,000 samples)"""
    samples = {}

    # Enumerate all combinations
    all_combos = []
    for (c1, c2), result in COLOR_MIXING.items():
        for template in TEMPLATES_1E:
            query = template.format(color1=c1, color2=c2)
            all_combos.append((query, result))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s1f_color_associations(num_samples: int = 3000) -> Dict[str, str]:
    """1F: Color Associations (3,000 samples)"""
    samples = {}

    # Enumerate all combinations
    all_combos = []
    for color, meanings in COLOR_ASSOCIATIONS.items():
        for meaning in meanings:
            for template in TEMPLATES_1F:
                query = template.format(meaning=meaning)
                all_combos.append((query, color))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


# ============================================================================
# STATEMENT 2: SHAPE PERCEPTION GENERATORS
# ============================================================================


def generate_s2a_object_shape_id(num_samples: int = 35000) -> Dict[str, str]:
    """2A: Object Shape Identification (35,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    # Combine 2D and 3D shapes
    all_shape_data = {**SHAPE_OBJECTS_2D, **SHAPE_OBJECTS_3D}
    all_shapes = list(all_shape_data.keys())

    while len(samples) < num_samples and attempt < max_attempts:
        shape = random.choice(all_shapes)
        obj = random.choice(all_shape_data[shape])

        # Randomly choose template type (60% base, 25% with adjective, 15% with context)
        template_type = random.random()

        if template_type < 0.60:
            # Base template
            template = random.choice(TEMPLATES_2A)
            query = template.format(obj=obj)
        elif template_type < 0.85:
            # Template with adjective
            adj = random.choice(ADJECTIVES)
            template = random.choice(TEMPLATES_2A_ADJ)
            query = template.format(adj=adj, obj=obj)
        else:
            # Template with context
            context = random.choice(CONTEXTS)
            template = random.choice(TEMPLATES_2A_CONTEXT)
            query = template.format(obj=obj, context=context)

        answer = shape

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s2b_reverse_shape_id(num_samples: int = 15000) -> Dict[str, str]:
    """2B: Reverse Shape Identification (15,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    # Combine 2D and 3D shapes
    all_shape_data = {**SHAPE_OBJECTS_2D, **SHAPE_OBJECTS_3D}
    all_shapes = list(all_shape_data.keys())

    # Shape descriptors
    shape_descriptors = {
        "circle": ["round", "circular"],
        "square": ["square"],
        "rectangle": ["rectangular"],
        "triangle": ["triangular"],
        "sphere": ["spherical", "round"],
        "cube": ["cubic"],
        "cylinder": ["cylindrical"],
        "cone": ["conical"],
    }

    while len(samples) < num_samples and attempt < max_attempts:
        shape = random.choice(all_shapes)
        descriptor = shape_descriptors.get(shape, [shape])[0]

        # Randomly choose template type (80% base, 20% with context)
        template_type = random.random()

        if template_type < 0.80:
            # Base template
            template = random.choice(TEMPLATES_2B)
            query = template.format(shape=descriptor)
        else:
            # Template with context
            context = random.choice(CONTEXTS)
            template = random.choice(TEMPLATES_2B_CONTEXT)
            query = template.format(shape=descriptor, context=context)

        answer = random.choice(all_shape_data[shape])

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s2c_shape_verification(num_samples: int = 8000) -> Dict[str, str]:
    """2C: Shape Verification (8,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    all_shape_data = {**SHAPE_OBJECTS_2D, **SHAPE_OBJECTS_3D}
    all_shapes = list(all_shape_data.keys())

    # 70% true, 30% false
    true_count = int(num_samples * 0.7)
    current_true = 0

    shape_descriptors = {
        "circle": ["round", "circular"],
        "square": ["square"],
        "rectangle": ["rectangular"],
        "triangle": ["triangular"],
        "sphere": ["spherical", "round"],
        "cube": ["cubic"],
        "cylinder": ["cylindrical"],
        "cone": ["conical"],
    }

    while len(samples) < num_samples and attempt < max_attempts:
        is_true = current_true < true_count

        if is_true:
            shape = random.choice(all_shapes)
            obj = random.choice(all_shape_data[shape])
            descriptor = shape_descriptors.get(shape, [shape])[0]
        else:
            shape = random.choice(all_shapes)
            wrong_shape = random.choice([s for s in all_shapes if s != shape])
            obj = random.choice(all_shape_data[wrong_shape])
            descriptor = shape_descriptors.get(shape, [shape])[0]

        template = random.choice(TEMPLATES_2C)
        query = template.format(
            obj=obj, obj_plural=pluralize_simple(obj), shape=descriptor
        )

        # Fix a/an for vowel-starting objects in templates that include an article.
        # (Uncountable nouns are not expected in shape objects, but handle defensively.)
        if is_uncountable(obj):
            query = query.replace(f" an {obj}", f" {obj}").replace(
                f" a {obj}", f" {obj}"
            )
            query = query.replace(f"An {obj}", obj.capitalize()).replace(
                f"A {obj}", obj.capitalize()
            )
        else:
            article = get_article(obj)
            if article == "an":
                query = query.replace(f"a {obj}", f"an {obj}")
            else:
                query = query.replace(f"an {obj}", f"a {obj}")
        answer = "yes" if is_true else "no"

        if query not in samples:
            samples[query] = answer
            if is_true:
                current_true += 1

        attempt += 1

    return samples


def generate_s2d_shape_multiple_choice(num_samples: int = 5000) -> Dict[str, str]:
    """2D: Shape Multiple Choice (5,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    all_shape_data = {**SHAPE_OBJECTS_2D, **SHAPE_OBJECTS_3D}
    all_shapes = list(all_shape_data.keys())

    shape_descriptors = {
        "circle": ["round", "circular"],
        "square": ["square"],
        "rectangle": ["rectangular"],
        "triangle": ["triangular"],
        "sphere": ["spherical", "round"],
        "cube": ["cubic"],
        "cylinder": ["cylindrical"],
        "cone": ["conical"],
    }

    while len(samples) < num_samples and attempt < max_attempts:
        shape = random.choice(all_shapes)
        obj1 = random.choice(all_shape_data[shape])

        other_shape = random.choice([s for s in all_shapes if s != shape])
        obj2 = random.choice(all_shape_data[other_shape])

        descriptor = shape_descriptors.get(shape, [shape])[0]

        # Randomize order
        if random.random() < 0.5:
            obj1, obj2 = obj2, obj1

        template = random.choice(TEMPLATES_2D)
        query = template.format(shape=descriptor, obj1=obj1, obj2=obj2)

        # Answer is the object that has the target shape
        answer = obj1 if get_shape_for_object(obj1) == shape else obj2

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s2e_2d_vs_3d(num_samples: int = 4000) -> Dict[str, str]:
    """2E: 2D vs 3D Distinction (4,000 samples)"""
    samples = {}

    # Enumerate all combinations
    all_combos = []
    for shape in ALL_SHAPES_2D:
        for template in TEMPLATES_2E:
            query = template.format(shape=shape, obj=shape)
            all_combos.append((query, "2D"))

    for shape in ALL_SHAPES_3D:
        for template in TEMPLATES_2E:
            query = template.format(shape=shape, obj=shape)
            all_combos.append((query, "3D"))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s2f_2d_3d_relationship(num_samples: int = 3000) -> Dict[str, str]:
    """2F: 2D-3D Relationship (3,000 samples)"""
    samples = {}

    # Enumerate all combinations
    all_combos = []
    for shape3d, shape2d in SHAPE_2D_3D.items():
        for template in TEMPLATES_2F_3D_TO_2D:
            query = template.format(shape3d=shape3d)
            all_combos.append((query, shape2d))

        for template in TEMPLATES_2F_2D_TO_3D:
            query = template.format(shape2d=shape2d)
            all_combos.append((query, shape3d))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


# ============================================================================
# STATEMENT 3: GEOMETRIC CONCEPTS GENERATORS
# ============================================================================


def generate_s3a_shape_by_sides(num_samples: int = 10000) -> Dict[str, str]:
    """3A: Shape by Number of Sides (10,000 samples)"""
    samples = {}

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for n_sides, shape_name in SIDES_TO_SHAPE.items():
        for template in TEMPLATES_3A:
            query = template.format(n=n_sides)
            all_combos.append((query, shape_name))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3b_sides_of_shape(num_samples: int = 10000) -> Dict[str, str]:
    """3B: Sides of Shape (10,000 samples)"""
    samples = {}

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for shape_name, n_sides in SHAPE_TO_SIDES.items():
        for template in TEMPLATES_3B:
            query = template.format(shape=shape_name)
            all_combos.append((query, str(n_sides)))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3c_corners_vertices(num_samples: int = 8000) -> Dict[str, str]:
    """3C: Corners/Vertices (8,000 samples)"""
    samples = {}

    # 2D shapes have vertices = sides
    # 3D shapes have specific vertex counts
    shape_vertices_2d = SHAPE_TO_SIDES.copy()
    shape_vertices_3d = {
        shape: props["vertices"] for shape, props in SHAPE_3D_PROPERTIES.items()
    }

    all_shape_vertices = {**shape_vertices_2d, **shape_vertices_3d}

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for shape_name, n_vertices in all_shape_vertices.items():
        for template in TEMPLATES_3C:
            query = template.format(shape=shape_name)
            all_combos.append((query, str(n_vertices)))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3d_angles_count(num_samples: int = 5000) -> Dict[str, str]:
    """3D: Angles Count (5,000 samples)"""
    samples = {}

    # 2D shapes have angles = sides
    shape_angles = SHAPE_TO_SIDES.copy()

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for shape_name, n_angles in shape_angles.items():
        for template in TEMPLATES_3D_ANGLES:
            query = template.format(shape=shape_name)
            all_combos.append((query, str(n_angles)))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3e_angle_types(num_samples: int = 8000) -> Dict[str, str]:
    """3E: Angle Types (8,000 samples)"""
    samples = {}

    # Enumerate combinations
    all_combos = []

    # Name queries
    for template in TEMPLATES_3E_NAME:
        if "less than 90" in template:
            all_combos.append((template, "acute"))
        elif "equal to 90" in template or "90 degrees called" in template:
            all_combos.append((template, "right angle"))
        elif "greater than 90" in template and "less than 180" in template:
            all_combos.append((template, "obtuse"))
        elif "equal to 180" in template or "180 degrees called" in template:
            all_combos.append((template, "straight angle"))
        elif "greater than 180" in template:
            all_combos.append((template, "reflex angle"))

    # Definition queries
    for template in TEMPLATES_3E_DEF:
        if "acute" in template:
            all_combos.append((template, "less than 90 degrees"))
        elif "right" in template:
            all_combos.append((template, "90"))
        elif "obtuse" in template:
            all_combos.append(
                (template, "greater than 90 degrees but less than 180 degrees")
            )
        elif "straight" in template:
            all_combos.append((template, "180 degrees"))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3f_shape_properties(num_samples: int = 8000) -> Dict[str, str]:
    """3F: Shape Properties (8,000 samples)"""
    samples = {}

    # Property descriptions
    properties = {
        "4 equal sides and 4 right angles": "square",
        "4 right angles": "rectangle",
        "4 equal sides": "rhombus",
        "2 pairs of parallel sides": "parallelogram",
        "exactly 1 pair of parallel sides": "trapezoid",
        "3 equal sides": "equilateral triangle",
        "2 equal sides": "isosceles triangle",
        "no equal sides": "scalene triangle",
        "one angle equals 90 degrees": "right triangle",
        "one angle greater than 90 degrees": "obtuse triangle",
    }

    # Enumerate combinations
    all_combos = []
    for prop, shape in properties.items():
        for template in TEMPLATES_3F:
            query = template.format(property=prop)
            all_combos.append((query, shape))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3g_3d_faces_edges_vertices(num_samples: int = 6000) -> Dict[str, str]:
    """3G: 3D Faces/Edges/Vertices (6,000 samples)"""
    samples = {}

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for shape3d, props in SHAPE_3D_PROPERTIES.items():
        for template in TEMPLATES_3G:
            query = template.format(shape3d=shape3d)

            # Determine which property is asked
            if "face" in template.lower() or "side" in template.lower():
                answer = str(props["faces"])
                all_combos.append((query, answer))
            elif "edge" in template.lower():
                answer = str(props["edges"])
                all_combos.append((query, answer))
            elif (
                "vert" in template.lower()
                or "corner" in template.lower()
                or "point" in template.lower()
            ):
                answer = str(props["vertices"])
                all_combos.append((query, answer))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3h_symmetry(num_samples: int = 3000) -> Dict[str, str]:
    """3H: Symmetry (3,000 samples)"""
    samples = {}

    # Enumerate combinations
    all_combos = []
    for shape, sym_count in SHAPE_SYMMETRY.items():
        for template in TEMPLATES_3H:
            query = template.format(shape=shape)
            all_combos.append((query, str(sym_count)))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3i_shape_comparisons(num_samples: int = 3000) -> Dict[str, str]:
    """3I: Shape Comparisons (3,000 samples)"""
    samples = {}

    shape_list = list(SHAPE_TO_SIDES.keys())

    if len(shape_list) < 2:
        return samples

    # Enumerate all combinations (avoids wasteful looping)
    all_combos = []
    for i, shape1 in enumerate(shape_list):
        for shape2 in shape_list[i + 1 :]:  # Only unique pairs
            sides1 = SHAPE_TO_SIDES[shape1]
            sides2 = SHAPE_TO_SIDES[shape2]

            for template in TEMPLATES_3I:
                query = template.format(shape1=shape1, shape2=shape2)

                # Determine answer based on template
                if "more" in template.lower():
                    answer = shape1 if sides1 > sides2 else shape2
                    all_combos.append((query, answer))
                elif "fewer" in template.lower() or "less" in template.lower():
                    answer = shape1 if sides1 < sides2 else shape2
                    all_combos.append((query, answer))

                # Also generate the reverse order pair
                query_rev = template.format(shape1=shape2, shape2=shape1)
                if "more" in template.lower():
                    answer_rev = shape2 if sides2 > sides1 else shape1
                    all_combos.append((query_rev, answer_rev))
                elif "fewer" in template.lower() or "less" in template.lower():
                    answer_rev = shape2 if sides2 < sides1 else shape1
                    all_combos.append((query_rev, answer_rev))

    # Shuffle and select up to num_samples
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3j_circle_properties(num_samples: int = 4000) -> Dict[str, str]:
    """3J: Circle Properties (4,000 samples)"""
    samples = {}

    # Enumerate combinations
    all_combos = []
    for desc, term in CIRCLE_PROPERTIES.items():
        for template in TEMPLATES_3J:
            if desc in template.lower() or any(
                word in template.lower() for word in desc.split()
            ):
                all_combos.append((template, term))

    # Manual mappings for specific templates
    specific_mappings = [
        ("What is the distance around a circle called?", "circumference"),
        ("What is the distance across a circle through the center called?", "diameter"),
        ("What is half of the diameter called?", "radius"),
        ("What do we call the distance from center to edge of a circle?", "radius"),
        ("What is the perimeter of a circle called?", "circumference"),
        (
            "What is a line from one side of a circle to the other through the center?",
            "diameter",
        ),
    ]

    all_combos.extend(specific_mappings)

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3k_basic_perimeter(num_samples: int = 2500) -> Dict[str, str]:
    """3K: Basic Perimeter (2,500 samples)"""
    samples = {}

    # Enumerate combinations
    all_combos = []

    # Square perimeters (side 1-15)
    for n in range(1, 16):
        for template in TEMPLATES_3K_SQUARE:
            query = template.format(n=n)
            answer = str(4 * n)
            all_combos.append((query, answer))

    # Rectangle perimeters
    for length in range(1, 13):
        for w in range(1, 11):
            if length != w:  # Avoid squares
                for template in TEMPLATES_3K_RECT:
                    query = template.format(l=length, w=w)
                    answer = str(2 * (length + w))
                    all_combos.append((query, answer))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s3l_basic_area(num_samples: int = 2500) -> Dict[str, str]:
    """3L: Basic Area (2,500 samples)"""
    samples = {}

    # Enumerate combinations
    all_combos = []

    # Square areas (side 1-12)
    for n in range(1, 13):
        for template in TEMPLATES_3L_SQUARE:
            query = template.format(n=n)
            answer = str(n * n)
            all_combos.append((query, answer))

    # Rectangle areas
    for length in range(1, 13):
        for w in range(1, 11):
            if length != w:  # Avoid squares
                for template in TEMPLATES_3L_RECT:
                    query = template.format(l=length, w=w)
                    answer = str(length * w)
                    all_combos.append((query, answer))

    # Shuffle and select
    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


# ============================================================================
# STATEMENT 4: PATTERN RECOGNITION GENERATORS
# ============================================================================


def generate_s4a_2item_alternating(num_samples: int = 8000) -> Dict[str, str]:
    """4A: 2-item Alternating Patterns (8,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    pattern_types = ["colors", "letters", "shapes", "numbers", "words"]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        ptype = random.choice(pattern_types)
        items = PATTERN_ITEMS[ptype]

        # Ensure we have enough items to sample
        sample_pool = items if len(items) > 10 else list(items)[:10]
        if len(sample_pool) < 2:
            attempt += 1
            continue

        item1, item2 = random.sample(sample_pool, 2)

        # Create pattern showing 4-6 repetitions
        reps = random.randint(2, 3)
        pattern_list = [item1, item2] * reps + [item1]
        pattern_str = format_pattern(pattern_list)

        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = str(item2)

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s4b_3item_repeating(num_samples: int = 7000) -> Dict[str, str]:
    """4B: 3-item Repeating Patterns (7,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    pattern_types = ["colors", "letters", "shapes", "numbers"]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        ptype = random.choice(pattern_types)
        items = PATTERN_ITEMS[ptype]

        # Ensure we have enough items to sample
        sample_pool = items if len(items) > 10 else list(items)[:10]
        if len(sample_pool) < 3:
            attempt += 1
            continue

        item1, item2, item3 = random.sample(sample_pool, 3)

        # Create pattern showing 2-3 full cycles
        reps = random.randint(2, 3)
        pattern_list = [item1, item2, item3] * reps
        # Add partial next cycle
        pattern_list.append(item1)

        pattern_str = format_pattern(pattern_list)

        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = str(item2)

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s4c_4item_repeating(num_samples: int = 4000) -> Dict[str, str]:
    """4C: 4-item Repeating Patterns (4,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    pattern_types = ["colors", "letters", "shapes", "days"]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        ptype = random.choice(pattern_types)
        items = PATTERN_ITEMS[ptype]

        if len(items) < 4:
            continue

        # Ensure we have enough items to sample
        sample_pool = items if len(items) > 10 else list(items)[:10]
        if len(sample_pool) < 4:
            attempt += 1
            continue

        item1, item2, item3, item4 = random.sample(sample_pool, 4)

        # Create pattern showing 2 full cycles
        pattern_list = [item1, item2, item3, item4] * 2
        # Add partial next cycle
        pattern_list.append(item1)

        pattern_str = format_pattern(pattern_list)

        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = str(item2)

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s4d_number_sequences(num_samples: int = 8000) -> Dict[str, str]:
    """4D: Simple Number Sequences (8,000 samples)"""
    samples = {}
    max_attempts = num_samples * 100
    attempt = 0

    # Sequence types: count by 1, 2, 3, 4, 5, 10
    increments = [1, 2, 3, 4, 5, 10, -1, -2]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        inc = random.choice(increments)
        start = random.randint(1, 50) if inc > 0 else random.randint(10, 50)

        # Generate 4-5 numbers
        length = random.randint(4, 5)
        pattern_list = [start + i * inc for i in range(length)]

        # Only keep if all positive and reasonable
        if all(n > 0 and n < 100 for n in pattern_list):
            answer_val = pattern_list[-1] + inc
            # Validate answer is also positive and reasonable
            if answer_val > 0 and answer_val < 100:
                pattern_str = format_pattern(pattern_list)
                template = random.choice(all_pattern_templates)
                query = template.format(pattern=pattern_str)
                answer = str(answer_val)

                if query not in samples:
                    samples[query] = answer

        attempt += 1

    return samples


def generate_s4e_growing_shrinking(num_samples: int = 4000) -> Dict[str, str]:
    """4E: Growing/Shrinking Patterns (4,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    base_items = ["A", "1", "X", "*", "O", "#", "@", "B", "C", "Z"]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        base = random.choice(base_items)

        # Create pattern A, AA, AAA, AAAA (with variations in starting length)
        start_len = random.randint(1, 2)
        pattern_list = [base * (start_len + i) for i in range(4)]
        pattern_str = format_pattern(pattern_list)

        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = base * (start_len + 4)

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s4f_doubling(num_samples: int = 3000) -> Dict[str, str]:
    """4F: Doubling Patterns (3,000 samples)"""
    samples = {}

    # Expanded start values
    starts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    all_combos = []
    for start in starts:
        # Doubling pattern
        pattern_list = [start * (2**i) for i in range(4)]
        if pattern_list[-1] * 2 < 1000:  # Keep answers reasonable
            pattern_str = format_pattern(pattern_list)

            for template in all_pattern_templates:
                query = template.format(pattern=pattern_str)
                answer = str(pattern_list[-1] * 2)
                all_combos.append((query, answer))

    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s4g_square_numbers(num_samples: int = 2000) -> Dict[str, str]:
    """4G: Square Numbers (2,000 samples)"""
    samples = {}

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    # Square sequences
    all_combos = []
    for start_idx in range(1, 12):  # Start from 1², 2², etc. (expanded range)
        pattern_list = [(start_idx + i) ** 2 for i in range(4)]
        pattern_str = format_pattern(pattern_list)

        for template in all_pattern_templates:
            query = template.format(pattern=pattern_str)
            answer = str((start_idx + 4) ** 2)
            all_combos.append((query, answer))

    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s4h_fibonacci(num_samples: int = 2000) -> Dict[str, str]:
    """4H: Fibonacci-like Patterns (2,000 samples)"""
    samples = {}

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    # Fibonacci sequences (expanded starting pairs)
    all_combos = []
    starts = [
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 1),
        (1, 3),
        (2, 3),
        (2, 5),
        (3, 5),
        (1, 4),
        (2, 4),
    ]

    for a, b in starts:
        fib = [a, b]
        for _ in range(4):
            fib.append(fib[-1] + fib[-2])

        pattern_list = fib[:5]
        pattern_str = format_pattern(pattern_list)

        for template in all_pattern_templates:
            query = template.format(pattern=pattern_str)
            answer = str(fib[5])
            all_combos.append((query, answer))

    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s4i_day_month_sequences(num_samples: int = 4000) -> Dict[str, str]:
    """4I: Day and Month Sequences (4,000 samples)"""
    samples = {}

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    all_combos = []

    # Day sequences
    days = PATTERN_ITEMS["days"]
    for i in range(len(days)):
        # Show 3-5 consecutive days (expanded range)
        for length in [3, 4, 5]:
            pattern_list = [days[(i + j) % len(days)] for j in range(length)]
            pattern_str = format_pattern(pattern_list)

            for template in all_pattern_templates:
                query = template.format(pattern=pattern_str)
                answer = days[(i + length) % len(days)]
                all_combos.append((query, answer))

    # Month sequences
    months = PATTERN_ITEMS["months"]
    for i in range(len(months)):
        # Show 3-5 consecutive months (expanded range)
        for length in [3, 4, 5]:
            pattern_list = [months[(i + j) % len(months)] for j in range(length)]
            pattern_str = format_pattern(pattern_list)

            for template in all_pattern_templates:
                query = template.format(pattern=pattern_str)
                answer = months[(i + length) % len(months)]
                all_combos.append((query, answer))

    random.shuffle(all_combos)
    for query, answer in all_combos[:num_samples]:
        if query not in samples:
            samples[query] = answer

    return samples


def generate_s4j_letter_sequences(num_samples: int = 4000) -> Dict[str, str]:
    """4J: Letter Sequences (4,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    letters = PATTERN_ITEMS["letters"]
    max_idx = len(letters) - 1  # 25 for 26 letters (0-indexed)

    while len(samples) < num_samples and attempt < max_attempts:
        # Choose sequence type: sequential, skip one, skip two, reverse
        seq_type = random.choice(["sequential", "skip_one", "skip_two", "reverse"])

        # Calculate valid start_idx range based on sequence type
        if seq_type == "sequential":
            max_start = max_idx - 4  # Need start + 4 to be valid
            if max_start < 0:
                attempt += 1
                continue
            start_idx = random.randint(0, max_start)
            max_idx_needed = start_idx + 4
            next_idx = start_idx + 4
        elif seq_type == "skip_one":
            max_start = max_idx - 8  # Need start + 8 to be valid
            if max_start < 0:
                attempt += 1
                continue
            start_idx = random.randint(0, max_start)
            max_idx_needed = start_idx + 6  # start + 3*2
            next_idx = start_idx + 8
        elif seq_type == "skip_two":
            max_start = max_idx - 12  # Need start + 12 to be valid
            if max_start < 0:
                attempt += 1
                continue
            start_idx = random.randint(0, max_start)
            max_idx_needed = start_idx + 9  # start + 3*3
            next_idx = start_idx + 12
        else:  # reverse
            max_start = max_idx - 4  # Need at least 4 letters from end
            if max_start < 0:
                attempt += 1
                continue
            start_idx = random.randint(0, max_start)
            max_idx_needed = max_idx - start_idx
            next_idx = max_idx - start_idx - 4

        # Check bounds before accessing
        if max_idx_needed > max_idx or next_idx > max_idx or next_idx < 0:
            attempt += 1
            continue

        # Now safe to access
        if seq_type == "sequential":
            pattern_list = [letters[start_idx + i] for i in range(4)]
            next_letter = letters[next_idx]
        elif seq_type == "skip_one":
            pattern_list = [letters[start_idx + i * 2] for i in range(4)]
            next_letter = letters[next_idx]
        elif seq_type == "skip_two":
            pattern_list = [letters[start_idx + i * 3] for i in range(4)]
            next_letter = letters[next_idx]
        else:  # reverse
            # Validate reverse pattern indices
            reverse_indices = [max_idx - start_idx - i for i in range(4)]
            if any(idx < 0 or idx > max_idx for idx in reverse_indices):
                attempt += 1
                continue
            pattern_list = [letters[idx] for idx in reverse_indices]
            next_letter = letters[next_idx]

        pattern_str = format_pattern(pattern_list)
        # Combine both template lists
        all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT
        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = next_letter

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


def generate_s4k_mixed_attributes(num_samples: int = 4000) -> Dict[str, str]:
    """4K: Mixed Attribute Patterns (4,000 samples)"""
    samples = {}
    max_attempts = num_samples * 50
    attempt = 0

    # Expanded attribute options
    sizes = PATTERN_ITEMS["sizes"]  # All sizes
    colors = PATTERN_ITEMS["colors"][:8]  # More colors
    shapes = PATTERN_ITEMS["shapes"][:6]  # More shapes

    # Combine both template lists
    all_pattern_templates = TEMPLATES_PATTERN + TEMPLATES_PATTERN_ALT

    while len(samples) < num_samples and attempt < max_attempts:
        # Choose attribute types
        attr_types = random.choice(
            [
                ("size", "color"),
                ("color", "shape"),
                ("size", "shape"),
            ]
        )

        if attr_types == ("size", "color"):
            vals1 = sizes
            vals2 = colors
        elif attr_types == ("color", "shape"):
            vals1 = colors
            vals2 = shapes
        else:  # size, shape
            vals1 = sizes
            vals2 = shapes

        # Ensure we have enough items
        if len(vals1) < 2 or len(vals2) < 2:
            attempt += 1
            continue

        # Create 2-item alternating pattern
        val1_1, val1_2 = random.sample(list(vals1), 2)
        val2_1, val2_2 = random.sample(list(vals2), 2)

        item1 = f"{val1_1} {val2_1}"
        item2 = f"{val1_2} {val2_2}"

        pattern_list = [item1, item2] * 2 + [item1]
        pattern_str = format_pattern(pattern_list)

        template = random.choice(all_pattern_templates)
        query = template.format(pattern=pattern_str)
        answer = item2

        if query not in samples:
            samples[query] = answer

        attempt += 1

    return samples


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================


def validate_distribution(all_samples: Dict[str, str]) -> None:
    """
    Validate that samples match expected distribution by categorizing them.
    """

    # Realistic targets based on combinatorial limits (quality over quantity)
    expected_counts = {
        "Statement 1: Color Perception": 55000,
        "Statement 2: Shape Perception": 40000,
        "Statement 3: Geometric Concepts": 10000,
        "Statement 4: Pattern Recognition": 45000,
    }
    # Total target: ~150,000 (vs original 250,000)

    categories = defaultdict(int)

    # Color association keywords (meanings that map to colors)
    color_association_keywords = [
        "love",
        "stop",
        "danger",
        "warning",
        "passion",
        "anger",
        "fire",
        "heat",
        "caution",
        "happiness",
        "joy",
        "sunshine",
        "optimism",
        "energy",
        "cheerful",
        "nature",
        "growth",
        "freshness",
        "environment",
        "health",
        "money",
        "luck",
        "calm",
        "peace",
        "trust",
        "loyalty",
        "sadness",
        "cold",
        "water",
        "sky",
        "royalty",
        "luxury",
        "mystery",
        "creativity",
        "wisdom",
        "spirituality",
        "femininity",
        "romance",
        "sweetness",
        "innocence",
        "youth",
        "softness",
        "warmth",
        "earth",
        "stability",
        "comfort",
        "reliability",
        "rustic",
        "elegance",
        "power",
        "sophistication",
        "darkness",
        "death",
        "evil",
        "purity",
        "cleanliness",
        "simplicity",
        "innocence",
        "surrender",
        "snow",
        "neutral",
        "balance",
        "maturity",
        "formal",
        "professional",
        "represents",
        "symbolizes",
        "associated with",
        "means",
        "stands for",
        "signifies",
        "denotes",
        "indicates",
        "conveys",
    ]

    # Color mixing keywords
    color_mixing_keywords = [
        "mix",
        "mixing",
        "blend",
        "blending",
        "combine",
        "combining",
        "plus",
        "together",
        "added",
        "result",
        "makes",
        "create",
    ]

    for query, answer in all_samples.items():
        query_lower = query.lower()

        # S4: Pattern Recognition - check first (most specific)
        # Check for pattern-specific phrases and also for "..." which indicates a sequence
        if (
            any(
                p in query_lower
                for p in [
                    "what comes next",
                    "complete the pattern",
                    "next item in the sequence",
                    "continue the pattern",
                    "find the next",
                    "what follows",
                    "next item:",
                    "continue:",
                    "next?",
                    "what's next",
                    "next term",
                    "the pattern",
                    "sequence:",
                    "series:",
                    "after",
                    "pattern:",
                    "extend the pattern",
                    "finish the sequence",
                    "look at this pattern",
                    "given the sequence",
                    "in the pattern",
                    "for the sequence",
                    "observe:",
                    "following",
                    "then next",
                    "starting with",
                    "if we have",
                ]
            )
            or ", ..." in query
        ):  # Patterns contain ", ..." like "red, blue, red, ..."
            categories["Statement 4: Pattern Recognition"] += 1

        # S3: Geometric Concepts - geometry terminology
        elif any(
            p in query_lower
            for p in [
                "sides",
                "vertices",
                "corners",
                "angles",
                "faces",
                "edges",
                "perimeter",
                "area",
                "symmetry",
                "degrees",
                "acute",
                "obtuse",
                "right angle",
                "diameter",
                "radius",
                "circumference",
                "polygon",
                "polyhedron",
                "-sided",
                "-gon",
                "property",
                "equal sides",
                "parallel",
                "right angles",
                "lines of symmetry",
            ]
        ):
            categories["Statement 3: Geometric Concepts"] += 1

        # S1: Color Perception - color keywords and associations
        elif (
            any(
                p in query_lower
                for p in [
                    "color",
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "brown",
                    "black",
                    "white",
                    "gray",
                    "grey",
                    "hue",
                    "shade",
                ]
            )
            or any(kw in query_lower for kw in color_association_keywords)
            or any(kw in query_lower for kw in color_mixing_keywords)
        ):
            categories["Statement 1: Color Perception"] += 1

        # S2: Shape Perception - shape keywords
        elif any(
            p in query_lower
            for p in [
                "shape",
                "circle",
                "square",
                "rectangle",
                "triangle",
                "sphere",
                "cube",
                "cylinder",
                "cone",
                "pyramid",
                "round",
                "circular",
                "rectangular",
                "triangular",
                "2d",
                "3d",
                "flat",
                "solid",
                "hexagon",
                "pentagon",
                "octagon",
                "oval",
                "diamond",
                "star",
                "heart",
                "prism",
                "oval-shaped",
                "star-shaped",
                "shaped like",
                "geometric",
                "outline",
                "form",
            ]
        ):
            categories["Statement 2: Shape Perception"] += 1

        else:
            categories["Uncategorized"] += 1
            # Debug: collect first 20 uncategorized samples
            if "uncategorized_samples" not in categories:
                categories["uncategorized_samples"] = []
            if len(categories["uncategorized_samples"]) < 20:
                categories["uncategorized_samples"].append((query, answer))

    # Print uncategorized samples for debugging
    if "uncategorized_samples" in categories and categories["uncategorized_samples"]:
        print("\n" + "=" * 80)
        print("SAMPLE UNCATEGORIZED QUERIES (for debugging):")
        print("=" * 80)
        for i, (q, a) in enumerate(categories["uncategorized_samples"][:20], 1):
            print(f"{i}. Q: {q[:80]}...")
            print(f"   A: {a}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("DISTRIBUTION VALIDATION")
    print("=" * 80)
    print(
        f"{'Category':<40} {'Actual':>10} {'Expected':>10} {'Difference':>12} {'Status':>10}"
    )
    print("-" * 80)

    has_issues = False
    for category in sorted(expected_counts.keys()):
        actual = categories.get(category, 0)
        expected = expected_counts[category]
        diff = actual - expected
        percent_diff = abs(diff) / expected * 100 if expected > 0 else 0

        # Tolerance: ±5% OK, ±10% WARNING, >10% ERROR
        if percent_diff <= 5.0:
            status = "✓ OK"
        elif percent_diff <= 10.0:
            status = "⚠ WARNING"
            has_issues = True
        else:
            status = "✗ ERROR"
            has_issues = True

        print(
            f"{category:<40} {actual:>10,} {expected:>10,} {diff:>+12,} ({percent_diff:>5.1f}%) {status:>10}"
        )

    # Show uncategorized if any
    uncategorized = categories.get("Uncategorized", 0)
    if uncategorized > 0:
        print(
            f"{'Uncategorized':<40} {uncategorized:>10,} {'0':>10} {'+' + str(uncategorized):>12} {'✗ ERROR':>10}"
        )
        has_issues = True

    print("-" * 80)
    total_categorized = sum(
        v for k, v in categories.items() if k != "Uncategorized" and isinstance(v, int)
    )
    print(
        f"{'TOTAL (categorized)':<40} {total_categorized:>10,} {sum(expected_counts.values()):>10,}"
    )

    if has_issues:
        print(
            "\n⚠ WARNING: Distribution has significant deviations from expected values!"
        )
        print("  This is normal if there are many duplicate queries across generators.")
    else:
        print("\n✓ Distribution looks good!")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Generate all samples and save to JSON."""
    print("Generating Group 3 Shapes, Colors & Patterns Dataset (250,000 samples)...")
    print("=" * 80)

    all_samples = {}
    before_count = 0

    # Statement 1: Color Perception (55,000 target - realistic)
    print("\n1. Generating Statement 1: Color Perception (55,000 target)...")

    print("   1A. Object Color ID (30,000)...")
    s1a = generate_s1a_object_color_id(30000)
    before_count = len(all_samples)
    all_samples.update(s1a)
    print(f"      Generated: {len(s1a)}, Added: {len(all_samples) - before_count}")

    print("   1B. Reverse Color ID (5,000)...")
    s1b = generate_s1b_reverse_color_id(5000)
    before_count = len(all_samples)
    all_samples.update(s1b)
    print(f"      Generated: {len(s1b)}, Added: {len(all_samples) - before_count}")

    print("   1C. Color Verification (8,000)...")
    s1c = generate_s1c_color_verification(8000)
    before_count = len(all_samples)
    all_samples.update(s1c)
    print(f"      Generated: {len(s1c)}, Added: {len(all_samples) - before_count}")

    print("   1D. Color Multiple Choice (5,000)...")
    s1d = generate_s1d_color_multiple_choice(5000)
    before_count = len(all_samples)
    all_samples.update(s1d)
    print(f"      Generated: {len(s1d)}, Added: {len(all_samples) - before_count}")

    print("   1E. Color Mixing (2,000)...")
    s1e = generate_s1e_color_mixing(2000)
    before_count = len(all_samples)
    all_samples.update(s1e)
    print(f"      Generated: {len(s1e)}, Added: {len(all_samples) - before_count}")

    print("   1F. Color Associations (2,500)...")
    s1f = generate_s1f_color_associations(2500)
    before_count = len(all_samples)
    all_samples.update(s1f)
    print(f"      Generated: {len(s1f)}, Added: {len(all_samples) - before_count}")

    # Statement 2: Shape Perception (40,000 target - realistic)
    print("\n2. Generating Statement 2: Shape Perception (40,000 target)...")

    print("   2A. Object Shape ID (25,000)...")
    s2a = generate_s2a_object_shape_id(25000)
    before_count = len(all_samples)
    all_samples.update(s2a)
    print(f"      Generated: {len(s2a)}, Added: {len(all_samples) - before_count}")

    print("   2B. Reverse Shape ID (5,000)...")
    s2b = generate_s2b_reverse_shape_id(5000)
    before_count = len(all_samples)
    all_samples.update(s2b)
    print(f"      Generated: {len(s2b)}, Added: {len(all_samples) - before_count}")

    print("   2C. Shape Verification (5,000)...")
    s2c = generate_s2c_shape_verification(5000)
    before_count = len(all_samples)
    all_samples.update(s2c)
    print(f"      Generated: {len(s2c)}, Added: {len(all_samples) - before_count}")

    print("   2D. Shape Multiple Choice (4,000)...")
    s2d = generate_s2d_shape_multiple_choice(4000)
    before_count = len(all_samples)
    all_samples.update(s2d)
    print(f"      Generated: {len(s2d)}, Added: {len(all_samples) - before_count}")

    print("   2E. 2D vs 3D (1,000)...")
    s2e = generate_s2e_2d_vs_3d(1000)
    before_count = len(all_samples)
    all_samples.update(s2e)
    print(f"      Generated: {len(s2e)}, Added: {len(all_samples) - before_count}")

    print("   2F. 2D-3D Relationship (500)...")
    s2f = generate_s2f_2d_3d_relationship(500)
    before_count = len(all_samples)
    all_samples.update(s2f)
    print(f"      Generated: {len(s2f)}, Added: {len(all_samples) - before_count}")

    # Statement 3: Geometric Concepts (10,000 target - limited by finite facts)
    print("\n3. Generating Statement 3: Geometric Concepts (10,000 target)...")

    print("   3A. Shape by Sides (500)...")
    s3a = generate_s3a_shape_by_sides(500)
    before_count = len(all_samples)
    all_samples.update(s3a)
    print(f"      Generated: {len(s3a)}, Added: {len(all_samples) - before_count}")

    print("   3B. Sides of Shape (600)...")
    s3b = generate_s3b_sides_of_shape(600)
    before_count = len(all_samples)
    all_samples.update(s3b)
    print(f"      Generated: {len(s3b)}, Added: {len(all_samples) - before_count}")

    print("   3C. Corners/Vertices (800)...")
    s3c = generate_s3c_corners_vertices(800)
    before_count = len(all_samples)
    all_samples.update(s3c)
    print(f"      Generated: {len(s3c)}, Added: {len(all_samples) - before_count}")

    print("   3D. Angles Count (500)...")
    s3d = generate_s3d_angles_count(500)
    before_count = len(all_samples)
    all_samples.update(s3d)
    print(f"      Generated: {len(s3d)}, Added: {len(all_samples) - before_count}")

    print("   3E. Angle Types (500)...")
    s3e = generate_s3e_angle_types(500)
    before_count = len(all_samples)
    all_samples.update(s3e)
    print(f"      Generated: {len(s3e)}, Added: {len(all_samples) - before_count}")

    print("   3F. Shape Properties (500)...")
    s3f = generate_s3f_shape_properties(500)
    before_count = len(all_samples)
    all_samples.update(s3f)
    print(f"      Generated: {len(s3f)}, Added: {len(all_samples) - before_count}")

    print("   3G. 3D Faces/Edges/Vertices (1,000)...")
    s3g = generate_s3g_3d_faces_edges_vertices(1000)
    before_count = len(all_samples)
    all_samples.update(s3g)
    print(f"      Generated: {len(s3g)}, Added: {len(all_samples) - before_count}")

    print("   3H. Symmetry (500)...")
    s3h = generate_s3h_symmetry(500)
    before_count = len(all_samples)
    all_samples.update(s3h)
    print(f"      Generated: {len(s3h)}, Added: {len(all_samples) - before_count}")

    print("   3I. Shape Comparisons (3,000)...")
    s3i = generate_s3i_shape_comparisons(3000)
    before_count = len(all_samples)
    all_samples.update(s3i)
    print(f"      Generated: {len(s3i)}, Added: {len(all_samples) - before_count}")

    print("   3J. Circle Properties (500)...")
    s3j = generate_s3j_circle_properties(500)
    before_count = len(all_samples)
    all_samples.update(s3j)
    print(f"      Generated: {len(s3j)}, Added: {len(all_samples) - before_count}")

    print("   3K. Basic Perimeter (1,000)...")
    s3k = generate_s3k_basic_perimeter(1000)
    before_count = len(all_samples)
    all_samples.update(s3k)
    print(f"      Generated: {len(s3k)}, Added: {len(all_samples) - before_count}")

    print("   3L. Basic Area (1,000)...")
    s3l = generate_s3l_basic_area(1000)
    before_count = len(all_samples)
    all_samples.update(s3l)
    print(f"      Generated: {len(s3l)}, Added: {len(all_samples) - before_count}")

    # Statement 4: Pattern Recognition (45,000 target - high combinatorial space)
    print("\n4. Generating Statement 4: Pattern Recognition (45,000 target)...")

    print("   4A. 2-item Alternating (7,000)...")
    s4a = generate_s4a_2item_alternating(7000)
    before_count = len(all_samples)
    all_samples.update(s4a)
    print(f"      Generated: {len(s4a)}, Added: {len(all_samples) - before_count}")

    print("   4B. 3-item Repeating (6,000)...")
    s4b = generate_s4b_3item_repeating(6000)
    before_count = len(all_samples)
    all_samples.update(s4b)
    print(f"      Generated: {len(s4b)}, Added: {len(all_samples) - before_count}")

    print("   4C. 4-item Repeating (4,000)...")
    s4c = generate_s4c_4item_repeating(4000)
    before_count = len(all_samples)
    all_samples.update(s4c)
    print(f"      Generated: {len(s4c)}, Added: {len(all_samples) - before_count}")

    print("   4D. Number Sequences (7,000)...")
    s4d = generate_s4d_number_sequences(7000)
    before_count = len(all_samples)
    all_samples.update(s4d)
    print(f"      Generated: {len(s4d)}, Added: {len(all_samples) - before_count}")

    print("   4E. Growing/Shrinking (4,000)...")
    s4e = generate_s4e_growing_shrinking(4000)
    before_count = len(all_samples)
    all_samples.update(s4e)
    print(f"      Generated: {len(s4e)}, Added: {len(all_samples) - before_count}")

    print("   4F. Doubling (3,000)...")
    s4f = generate_s4f_doubling(3000)
    before_count = len(all_samples)
    all_samples.update(s4f)
    print(f"      Generated: {len(s4f)}, Added: {len(all_samples) - before_count}")

    print("   4G. Square Numbers (2,000)...")
    s4g = generate_s4g_square_numbers(2000)
    before_count = len(all_samples)
    all_samples.update(s4g)
    print(f"      Generated: {len(s4g)}, Added: {len(all_samples) - before_count}")

    print("   4H. Fibonacci (2,000)...")
    s4h = generate_s4h_fibonacci(2000)
    before_count = len(all_samples)
    all_samples.update(s4h)
    print(f"      Generated: {len(s4h)}, Added: {len(all_samples) - before_count}")

    print("   4I. Day/Month Sequences (4,000)...")
    s4i = generate_s4i_day_month_sequences(4000)
    before_count = len(all_samples)
    all_samples.update(s4i)
    print(f"      Generated: {len(s4i)}, Added: {len(all_samples) - before_count}")

    print("   4J. Letter Sequences (3,000)...")
    s4j = generate_s4j_letter_sequences(3000)
    before_count = len(all_samples)
    all_samples.update(s4j)
    print(f"      Generated: {len(s4j)}, Added: {len(all_samples) - before_count}")

    print("   4K. Mixed Attributes (3,000)...")
    s4k = generate_s4k_mixed_attributes(3000)
    before_count = len(all_samples)
    all_samples.update(s4k)
    print(f"      Generated: {len(s4k)}, Added: {len(all_samples) - before_count}")

    # Validate distribution
    validate_distribution(all_samples)

    # Save to TXT (in curriculum_training_data/output folder)
    script_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "group3.txt")
    print(f"\n{'=' * 80}")

    # Curriculum progression order:
    # - Patterns first (Statement 4)
    # - Then colors (Statement 1), then shapes (Statement 2), then geometry (Statement 3)
    ordered_qa_pairs_dict = {}

    # Collect QA pairs in curriculum order and deduplicate
    for sample_dict in [
        s4a,
        s4b,
        s4c,
        s4d,
        s4e,
        s4f,
        s4g,
        s4h,
        s4i,
        s4j,
        s4k,
        s1a,
        s1b,
        s1c,
        s1d,
        s1e,
        s1f,
        s2a,
        s2b,
        s2c,
        s2d,
        s2e,
        s2f,
        s3a,
        s3b,
        s3c,
        s3d,
        s3e,
        s3f,
        s3g,
        s3h,
        s3i,
        s3j,
        s3k,
        s3l,
    ]:
        for query, answer in sample_dict.items():
            finalized_query = _finalize_group3_prompt(query)
            # Deduplicate: if same finalized query exists, keep the first one
            if finalized_query not in ordered_qa_pairs_dict:
                ordered_qa_pairs_dict[finalized_query] = answer

    # Convert to list maintaining curriculum order
    ordered_qa_pairs = [(q, a) for q, a in ordered_qa_pairs_dict.items()]

    # Combine QA pairs into samples where all questions have answers
    # Format: "Q1? A1. Q2? A2. Q3? A3. ..." until reaching 512 tokens per sample
    print(f"\n{'=' * 80}")
    print("Combining QA pairs into samples (all questions with answers)...")
    print("  Target: >= 512 tokens per sample")
    combined_samples = combine_qa_pairs_to_reach_min_tokens(
        ordered_qa_pairs, min_tokens=512
    )
    print(f"  Original QA pairs: {len(ordered_qa_pairs):,}")
    print(f"  Combined samples: {len(combined_samples):,}")

    print(f"\nSaving {len(combined_samples):,} samples to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in combined_samples:
            f.write(sample + "\n")

    print(f"\n✓ Successfully saved {len(combined_samples):,} samples!")
    print("\nValidation Summary:")
    print(f"  - Total samples (after combining): {len(combined_samples):,}")
    print("  - Expected: 250,000")
    print(f"  - Difference: {len(ordered_qa_pairs) - 250000:,}")

    if abs(len(ordered_qa_pairs) - 250000) / 250000 <= 0.10:
        print("\n✓ Sample count is within expected range!")
    else:
        print("\n⚠ Note: Sample count differs from expected.")


if __name__ == "__main__":
    main()
