# Garbage Token Fix Summary: Dataset Preprocessing

This report identifies garbage tokens that should be addressed via **Dataset Preprocessing** to prevent them from entering the training pipeline or to clean existing noise.

| Category | Count | Decoded Values (Unicode) | Impacted Token IDs | Suggested Fix & Target File |
| :--- | :--- | :--- | :--- | :--- |
| **`zero_width`*** | 20 | `ZWSP` (\u200b), `BOM` (\ufeff), `Bidi` (\u202a-\u202e), `Word Joiner` (\u2060) | 2787, 5820, 10363, 10417, 14642, 18982, 19568, 19836, 22545, 49176, 55932, 80203, 86647, 89190, 90817, 96843, 102216, 104890, 107765, 110669 | **Fix:** Use regex to filter non-essential invisible characters (excluding ZWJ/ZWNJ).<br>**Target:** `dataset_cleaning_script.py` (Pre-tokenization stage) |
| **`html_artifact`** | 4 | `&#`, ` &#`, ` '&#`, `;&#` (Unescaped Entities) | 18635, 42631, 45060, 101607 | **Fix:** Apply `html.unescape()` to correctly resolve encoded entities back to plain text.<br>**Target:** `dataset_cleaning_script.py` (Text cleaning pipeline) |

*\*Note: The `zero_width` category excludes ZWJ/ZWNJ which are legitimate for Indic script shaping.*
