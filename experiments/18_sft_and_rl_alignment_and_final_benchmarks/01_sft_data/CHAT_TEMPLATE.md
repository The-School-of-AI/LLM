# Chat Template — Team 18 SFT

Use **one** of the following templates consistently for all SFT data and evaluation. The base model / tokenizer (Team 6) determines which is applicable.

---

## Option A: ChatML (OpenAI-style)

Common for models that use `<|im_start|>` and `<|im_end|>`.

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

- **Special tokens:** `<|im_start|>`, `<|im_end|>`, and optionally `<|endoftext|>`.
- Ensure these are in the tokenizer; if not, add via `add_special_tokens` (coordinate with Team 6).

---

## Option B: Llama 3 / Llama 3.1

Meta’s standard for Llama 3 family.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_message}<|eot_id|>
```

- **Special tokens:** `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`.
- Usually already in the Llama tokenizer.

---

## Option C: Custom (document here)

If you use a custom template (e.g. for an Eleuther or in-house base), document the exact format and all special tokens below:

- **Template string or pattern:**
- **Special tokens added:**
- **Tokenizer compatibility:**

---

## Implementation

- Apply the chosen template in `scripts/apply_chat_template.py`.
- Use the same template in the training data collator and in evaluation (Team 17 benchmarks).
- **Do not** tune prompts per benchmark; use this template only.
