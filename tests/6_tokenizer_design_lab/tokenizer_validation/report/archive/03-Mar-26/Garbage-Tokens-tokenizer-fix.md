# Garbage Token Fix Summary: Tokenizer Pruning

This report identifies garbage tokens that should be addressed via **Tokenizer Construction** logic to prune inherited noise from the base vocabulary.

| Category | Count | Decoded Values (Unicode) | Impacted Token IDs | Suggested Fix & Target File |
| :--- | :--- | :--- | :--- | :--- |
| **`broken_utf8`** | 20 | ``, `s`, `n`, etc. (Literal Replacement Chars) | 2740, 8100, 21607, 21812, 47472, 51441, 54350, 55055, 61113, 63696, 68156, 79325, 80851, 88806, 98892, 101501, 106278, 113213, 113903, 114227 | **Fix:** Explicitly prune `U+FFFD` containing tokens during vocabulary construction.<br>**Target:** `Team_6_File\LLM\experiments\6_tokenizer_design_lab\build_clean_tokenizer.py` (Step 3: Removal logic) |
| **`private_use`** | 5 | `\uf0b7`, `\ue934`, `\uf0a7`, `\uf0d8`, etc. (PUA Artifacts) | 49529, 76039, 77811, 99490, 111144 | **Fix:** Add Private Use Area (PUA) codepoint ranges (`0xE000-0xF8FF`) to the blocked script list.<br>**Target:** `Team_6_File\LLM\experiments\6_tokenizer_design_lab\build_clean_tokenizer.py` (Lines 68-106: `BLOCKED_RANGES`) |

*\*Note: Implementation in the tokenizer script is the most direct way to ensure a clean audit report.*
