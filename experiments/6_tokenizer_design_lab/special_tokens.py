"""Special tokens definitions for the tokenizer.

This module contains all special token definitions organized by category.
These are imported by build_clean_tokenizer.py to build the final tokenizer.
"""

# Document and control tokens
DOC_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|pad|>",
    "<|unk|>",
    "<|sep|>",
    "<|cls|>",
    "<|mask|>",
    "<|newline|>",
    "<|paragraph|>",
    "<|document|>",
]

# Chat/conversation role tokens
CHAT_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|function|>",
    "<|context|>",
    "<|instruction|>",
    "<|response|>",
    "<|turn|>",
    "<|end_turn|>",
]

# Code block tokens
CODE_TOKENS = [
    "<|code_begin|>",
    "<|code_end|>",
    "<|output_begin|>",
    "<|output_end|>",
    "<|error|>",
    "<|stdin|>",
    "<|stdout|>",
    "<|stderr|>",
    "<|file_begin|>",
    "<|file_end|>",
]

# Programming language tags
LANG_TOKENS = [
    "<|lang:python|>",
    "<|lang:javascript|>",
    "<|lang:typescript|>",
    "<|lang:java|>",
    "<|lang:cpp|>",
    "<|lang:c|>",
    "<|lang:rust|>",
    "<|lang:go|>",
    "<|lang:ruby|>",
    "<|lang:php|>",
    "<|lang:swift|>",
    "<|lang:kotlin|>",
    "<|lang:scala|>",
    "<|lang:r|>",
    "<|lang:julia|>",
    "<|lang:sql|>",
    "<|lang:html|>",
    "<|lang:css|>",
    "<|lang:bash|>",
    "<|lang:shell|>",
]

# Tool/function calling tokens
TOOL_TOKENS = [
    "<|json_begin|>",
    "<|json_end|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|function_call|>",
    "<|function_result|>",
    "<|api_request|>",
    "<|api_response|>",
    "<|schema|>",
    "<|arguments|>",
]

# Data source attribution tokens
SOURCE_TOKENS = [
    "<|source:wikipedia|>",
    "<|source:github|>",
    "<|source:arxiv|>",
    "<|source:web|>",
    "<|source:book|>",
    "<|source:code|>",
    "<|source:docs|>",
    "<|source:news|>",
    "<|source:social|>",
    "<|source:other|>",
]

# Thinking/reasoning tokens
THINK_TOKENS = [
    "<|think_begin|>",
    "<|think_end|>",
    "<|step|>",
    "<|plan|>",
    "<|reflect|>",
    "<|verify|>",
    "<|conclude|>",
    "<|reason|>",
    "<|analyze|>",
    "<|synthesize|>",
]

# FIM (Fill-in-the-Middle) tokens for code completion
FIM_TOKENS = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
]

# Vision/Multimodal tokens
VISION_TOKENS = [
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

# Object reference tokens (for grounding)
OBJECT_REF_TOKENS = [
    "<|object_ref_start|>",
    "<|object_ref_end|>",
]

# Bounding box/quad tokens (for visual grounding)
BOX_TOKENS = [
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
]

# Chat format tokens (im = instruction/message)
IM_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
]

# Code context tokens
CODE_CONTEXT_TOKENS = [
    "<|file_sep|>",
    "<|repo_name|>",
]

# XML-style tool use tokens (from Qwen)
XML_TOOL_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]

# Thinking tokens (for reasoning models)
REASONING_TOKENS = [
    "<think>",
    "</think>",
]

# End of turn token
EOT_TOKENS = [
    "<|EOT|>",
]

# Combined list of all base special tokens (first 256 slots)
BASE_SPECIAL_TOKENS = (
    DOC_TOKENS
    + CHAT_TOKENS
    + CODE_TOKENS
    + LANG_TOKENS
    + TOOL_TOKENS
    + SOURCE_TOKENS
    + THINK_TOKENS
)

# Additional special tokens added at the end of vocabulary
# These are from Qwen-Code and DeepSeek-Code tokenizers
ADDITIONAL_SPECIAL_TOKENS = (
    FIM_TOKENS
    + VISION_TOKENS
    + OBJECT_REF_TOKENS
    + BOX_TOKENS
    + IM_TOKENS
    + CODE_CONTEXT_TOKENS
    + XML_TOOL_TOKENS
    + REASONING_TOKENS
    + EOT_TOKENS
)

# Total count of additional tokens
NUM_ADDITIONAL_SPECIAL = len(ADDITIONAL_SPECIAL_TOKENS)  # 26
