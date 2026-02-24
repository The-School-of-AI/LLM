"""Comprehensive audit of tsai_131k_tokenizer."""
import json
from collections import defaultdict

PATH = "../tsai_131k_tokenizer/tokenizer.json"

def _build_gpt2_maps():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs = list(bs); n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_C2B = _build_gpt2_maps()

def dec(t):
    try: return bytes(_C2B[c] for c in t).decode("utf-8", errors="replace")
    except KeyError: return t

INDIC = [(0x0900,0x097F,"Devanagari"),(0xA8E0,0xA8FF,"Devanagari Ext"),
    (0x0980,0x09FF,"Bengali"),(0x0A00,0x0A7F,"Gurmukhi"),(0x0A80,0x0AFF,"Gujarati"),
    (0x0B00,0x0B7F,"Oriya"),(0x0B80,0x0BFF,"Tamil"),(0x0C00,0x0C7F,"Telugu"),
    (0x0C80,0x0CFF,"Kannada"),(0x0D00,0x0D7F,"Malayalam"),(0x0D80,0x0DFF,"Sinhala")]

BLOCKED = [(0x0370,0x03FF,"Greek"),(0x0400,0x04FF,"Cyrillic"),(0x0500,0x052F,"Cyrillic Sup"),
    (0x0530,0x058F,"Armenian"),(0x0590,0x05FF,"Hebrew"),(0x0600,0x06FF,"Arabic"),
    (0x0750,0x077F,"Arabic Sup"),(0x0700,0x074F,"Syriac"),(0x0780,0x07BF,"Thaana"),
    (0x0E00,0x0E7F,"Thai"),(0x0E80,0x0EFF,"Lao"),(0x1780,0x17FF,"Khmer"),
    (0x0F00,0x0FFF,"Tibetan"),(0x1800,0x18AF,"Mongolian"),
    (0x3000,0x303F,"CJK Sym"),(0x3040,0x309F,"Hiragana"),(0x30A0,0x30FF,"Katakana"),
    (0x3400,0x4DBF,"CJK ExtA"),(0x4E00,0x9FFF,"CJK Unified"),
    (0xAC00,0xD7AF,"Hangul"),(0x1100,0x11FF,"Hangul Jamo"),
    (0x1200,0x137F,"Ethiopic"),(0x1000,0x109F,"Myanmar"),
    (0xFB50,0xFDFF,"Arabic PrA"),(0xFE70,0xFEFF,"Arabic PrB"),
    (0x20000,0x2A6DF,"CJK ExtB")]

def has_blocked(t):
    for ch in dec(t):
        cp = ord(ch)
        for lo,hi,nm in BLOCKED:
            if lo<=cp<=hi: return nm
    return None

def indic_scripts(t):
    s = set()
    for ch in dec(t):
        cp = ord(ch)
        for lo,hi,nm in INDIC:
            if lo<=cp<=hi: s.add(nm)
    return s

def is_indic(t):
    return len(indic_scripts(t)) > 0

# Build pipe-delimited special token names to search for
def make_special(name):
    """Build token like <|name|>"""
    return "<" + "|" + name + "|" + ">"

def make_xml(name):
    """Build token like <name>"""
    return "<" + name + ">"

def make_xml_close(name):
    """Build token like </name>"""
    return "<" + "/" + name + ">"

def main():
    out = []
    def P(s=""): out.append(s); print(s)

    P("=" * 80)
    P("         TOKENIZER AUDIT REPORT: tsai_131k_tokenizer")
    P("=" * 80)

    with open(PATH, encoding="utf-8") as f:
        tok = json.load(f)

    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    added = tok.get("added_tokens", [])

    P(f"\nVocab size: {len(vocab):,}")
    P(f"Merges: {len(merges):,}")
    P(f"Added tokens (special): {len(added)}")

    # ─── CHECK 1: Vocab size ───
    P("\n" + "-"*60)
    P("CHECK 1: VOCAB SIZE == 131,072")
    P("-"*60)
    if len(vocab) == 131072:
        P("  [PASS] 131,072 (2^17)")
    else:
        P(f"  [FAIL] {len(vocab):,} != 131,072")

    # ─── CHECK 2: Required special tokens ───
    P("\n" + "-"*60)
    P("CHECK 2: REQUIRED SPECIAL TOKENS FOR PRE/POST TRAINING")
    P("-"*60)

    added_set = {t["content"] for t in added}

    required_pipe = [
        ("BOS", "begin_of_text"),
        ("EOS", "end_of_text"),
        ("PAD", "pad"),
        ("UNK", "unk"),
        ("SEP", "sep"),
        ("CLS", "cls"),
        ("MASK", "mask"),
        ("SYSTEM role", "system"),
        ("USER role", "user"),
        ("ASSISTANT role", "assistant"),
        ("TOOL role", "tool"),
        ("FUNCTION role", "function"),
        ("CONTEXT", "context"),
        ("INSTRUCTION", "instruction"),
        ("RESPONSE", "response"),
        ("TURN marker", "turn"),
        ("END_TURN", "end_turn"),
        ("EOT", "EOT"),
        # Code
        ("CODE_BEGIN", "code_begin"),
        ("CODE_END", "code_end"),
        ("OUTPUT_BEGIN", "output_begin"),
        ("OUTPUT_END", "output_end"),
        ("ERROR", "error"),
        # FIM
        ("FIM_PREFIX", "fim_prefix"),
        ("FIM_MIDDLE", "fim_middle"),
        ("FIM_SUFFIX", "fim_suffix"),
        ("FIM_PAD", "fim_pad"),
        # Vision
        ("VISION_START", "vision_start"),
        ("VISION_END", "vision_end"),
        ("IMAGE_PAD", "image_pad"),
        # Tool calling
        ("TOOL_CALL", "tool_call"),
        ("TOOL_RESULT", "tool_result"),
        ("FUNCTION_CALL", "function_call"),
        ("FUNCTION_RESULT", "function_result"),
        ("JSON_BEGIN", "json_begin"),
        ("JSON_END", "json_end"),
        ("API_REQUEST", "api_request"),
        ("API_RESPONSE", "api_response"),
        ("SCHEMA", "schema"),
        ("ARGUMENTS", "arguments"),
        # Thinking / reasoning
        ("THINK_BEGIN", "think_begin"),
        ("THINK_END", "think_end"),
        # Chat format
        ("IM_START", "im_start"),
        ("IM_END", "im_end"),
        # Code context
        ("FILE_SEP", "file_sep"),
        ("REPO_NAME", "repo_name"),
    ]

    required_xml = [
        ("XML tool_call open", "tool_call"),
        ("XML tool_call close", "/tool_call"),
        ("XML tool_response open", "tool_response"),
        ("XML tool_response close", "/tool_response"),
        ("XML think open", "think"),
        ("XML think close", "/think"),
    ]

    all_pass = True
    for label, name in required_pipe:
        token = make_special(name)
        found = token in added_set or token in vocab
        status = "[PASS]" if found else "[FAIL]"
        if not found: all_pass = False
        # Also print the ID if found
        tid = vocab.get(token, "N/A")
        P(f"  {status} {label:25s} -> {token:30s} (ID: {tid})")

    for label, name in required_xml:
        token = "<" + name + ">"
        found = token in added_set or token in vocab
        status = "[PASS]" if found else "[FAIL]"
        if not found: all_pass = False
        tid = vocab.get(token, "N/A")
        P(f"  {status} {label:25s} -> {token:30s} (ID: {tid})")

    P(f"\n  Overall: {'ALL PRESENT' if all_pass else 'SOME MISSING'}")

    # ─── CHECK 2b: Agent token format check ───
    P("\n" + "-"*60)
    P("CHECK 2b: AGENT TOKEN FORMAT (supervisor asked about <AGENT> vs |AGENT|)")
    P("-"*60)
    agent_variants = [
        make_special("agent"),
        "<AGENT>",
        "|AGENT|",
        "AGENT",
        make_special("AGENT"),
    ]
    for av in agent_variants:
        found_in_vocab = av in vocab
        found_in_added = av in added_set
        P(f"  '{av}': in_vocab={found_in_vocab}, in_added={found_in_added}")

    # Check if assistant is used instead of agent
    P(f"\n  Note: The tokenizer uses {make_special('assistant')} for the agent role")
    P(f"        {make_special('assistant')} in vocab: {make_special('assistant') in vocab}")
    P(f"        {make_special('assistant')} in added: {make_special('assistant') in added_set}")

    # ─── CHECK 3: Reserved tokens ───
    P("\n" + "-"*60)
    P("CHECK 3: RESERVED TOKENS (250 expected)")
    P("-"*60)
    reserved_count = 0
    reserved_ids = []
    for t in added:
        if "reserved_" in t["content"]:
            reserved_count += 1
            reserved_ids.append(t["id"])
    P(f"  Found {reserved_count} reserved tokens")
    if reserved_count == 250:
        P("  [PASS] Exactly 250 reserved tokens")
    else:
        P(f"  [FAIL] Expected 250, got {reserved_count}")
    if reserved_ids:
        P(f"  ID range: {min(reserved_ids)} to {max(reserved_ids)}")

    # ─── CHECK 4: Language tag tokens ───
    P("\n" + "-"*60)
    P("CHECK 4: PROGRAMMING LANGUAGE TAGS")
    P("-"*60)
    langs = ["python","javascript","typescript","java","cpp","c","rust","go",
             "ruby","php","swift","kotlin","scala","r","julia","sql","html","css","bash","shell"]
    for lang in langs:
        token = make_special(f"lang:{lang}")
        found = token in added_set or token in vocab
        status = "[PASS]" if found else "[FAIL]"
        P(f"  {status} {token}")

    # ─── CHECK 5: No tokens > 32 chars ───
    P("\n" + "-"*60)
    P("CHECK 5: NO TOKENS > 32 CHARACTERS (excluding special)")
    P("-"*60)
    long_tokens = []
    for t in vocab:
        if t in added_set:
            continue
        if len(t) > 32:
            long_tokens.append((t, len(t), vocab[t]))
    P(f"  Tokens > 32 chars: {len(long_tokens)}")
    if len(long_tokens) == 0:
        P("  [PASS] No tokens exceed 32 characters")
    else:
        P("  [FAIL] The following tokens exceed 32 chars:")
        for t, tok_len, tid in long_tokens[:20]:
            P(f"    ID {tid}: len={tok_len} '{dec(t)[:50]}'")
        if len(long_tokens) > 20:
            P(f"    ... and {len(long_tokens)-20} more")

    # ─── CHECK 6: No blocked scripts ───
    P("\n" + "-"*60)
    P("CHECK 6: NO BLOCKED SCRIPTS (CJK, Arabic, Cyrillic, etc.)")
    P("-"*60)
    blocked_found = defaultdict(list)
    for t in vocab:
        if t in added_set:
            continue
        bl = has_blocked(t)
        if bl:
            blocked_found[bl].append(t)
    if not blocked_found:
        P("  [PASS] No blocked script tokens found")
    else:
        P(f"  [FAIL] Found {sum(len(v) for v in blocked_found.values())} blocked tokens:")
        for script, tokens in sorted(blocked_found.items(), key=lambda x: -len(x[1])):
            P(f"    {script}: {len(tokens)} tokens")
            for t in tokens[:3]:
                P(f"      e.g. ID {vocab[t]}: '{dec(t)[:40]}'")

    # ─── CHECK 7: Indic language coverage ───
    P("\n" + "-"*60)
    P("CHECK 7: INDIC LANGUAGE COVERAGE")
    P("-"*60)
    indic_by_script = defaultdict(list)
    for t in vocab:
        if t in added_set:
            continue
        scripts = indic_scripts(t)
        for s in scripts:
            indic_by_script[s].append(t)

    expected_scripts = ["Devanagari","Bengali","Tamil","Telugu","Kannada","Malayalam","Gujarati","Gurmukhi","Oriya","Sinhala"]
    total_indic = 0
    for script in expected_scripts:
        count = len(indic_by_script.get(script, []))
        total_indic += count
        status = "[PASS]" if count > 0 else "[FAIL]"
        P(f"  {status} {script:20s}: {count:,} tokens")
        # Show a few samples
        if count > 0:
            samples = indic_by_script[script][:5]
            P(f"         Samples: {[dec(s) for s in samples]}")

    P(f"\n  Total Indic tokens: {total_indic:,}")

    # ─── CHECK 8: Code optimization ───
    P("\n" + "-"*60)
    P("CHECK 8: CODE-OPTIMIZED TOKENS (Python, JS/TS, C/C++, configs)")
    P("-"*60)

    py_kw = {"def","class","import","from","return","yield","async","await","self",
             "__init__","__main__","__name__","print","lambda","None","True","False"}
    js_kw = {"function","const","let","var","export","require","module","console",
             "document","window","undefined","typeof","Promise","async","await"}
    c_kw = {"struct","typedef","sizeof","include","define","ifdef","ifndef","endif",
            "printf","malloc","free","NULL","static","extern","unsigned","enum","void"}

    code_tokens = {"python":[], "javascript":[], "c_cpp":[], "json_struct":[], "indentation":[]}

    for t in vocab:
        if t in added_set: continue
        d = dec(t)
        ds = d.strip()

        # Check Python
        for kw in py_kw:
            if kw in ds: code_tokens["python"].append(t); break

        # Check JS
        for kw in js_kw:
            if kw in ds: code_tokens["javascript"].append(t); break

        # Check C/C++
        for kw in c_kw:
            if kw in ds: code_tokens["c_cpp"].append(t); break

        # Check JSON structural
        if any(p in d for p in ['{"', '"}', '": ', '["', '"]', 'true', 'false', 'null']):
            code_tokens["json_struct"].append(t)

        # Indentation tokens (spaces/tabs at start)
        if d and all(c in ' \t' for c in d) and len(d) >= 2:
            code_tokens["indentation"].append(t)

    for cat, toks in code_tokens.items():
        P(f"  {cat:20s}: {len(toks):,} tokens")
        if toks:
            samples = toks[:5]
            P(f"    Samples: {[repr(dec(s)) for s in samples]}")

    # ─── CHECK 9: No garbage tokens ───
    P("\n" + "-"*60)
    P("CHECK 9: GARBAGE TOKEN SCAN (suspiciously useless tokens)")
    P("-"*60)

    garbage_candidates = []
    for t in vocab:
        if t in added_set: continue
        d = dec(t)
        # Check for replacement chars
        if "\ufffd" in d and len(d) > 1:
            garbage_candidates.append((t, vocab[t], d, "contains replacement char"))
        # Check for only control chars (excluding whitespace)
        if d and all(ord(c) < 32 and c not in '\n\r\t ' for c in d):
            garbage_candidates.append((t, vocab[t], repr(d), "control chars only"))

    P(f"  Suspicious tokens found: {len(garbage_candidates)}")
    if garbage_candidates:
        for t, tid, d, reason in garbage_candidates[:20]:
            P(f"    ID {tid}: '{d[:40]}' ({reason})")
    else:
        P("  [PASS] No obvious garbage tokens detected")

    # ─── CHECK 10: Token ID layout ───
    P("\n" + "-"*60)
    P("CHECK 10: TOKEN ID LAYOUT")
    P("-"*60)

    # Find boundaries
    special_ids = sorted([t["id"] for t in added])
    non_special_ids = sorted([v for k,v in vocab.items() if k not in added_set])
    indic_ids = sorted([v for k,v in vocab.items() if k not in added_set and is_indic(k)])
    general_ids = sorted([v for k,v in vocab.items() if k not in added_set and not is_indic(k)])

    P(f"  Non-special tokens: {len(non_special_ids):,}")
    if non_special_ids:
        P(f"    ID range: {non_special_ids[0]} to {non_special_ids[-1]}")
    P(f"  General tokens: {len(general_ids):,}")
    if general_ids:
        P(f"    ID range: {general_ids[0]} to {general_ids[-1]}")
    P(f"  Indic tokens: {len(indic_ids):,}")
    if indic_ids:
        P(f"    ID range: {indic_ids[0]} to {indic_ids[-1]}")
    P(f"  Special tokens: {len(special_ids)}")
    if special_ids:
        P(f"    ID range: {special_ids[0]} to {special_ids[-1]}")

    # Verify layout: general < indic < special
    if general_ids and indic_ids and special_ids:
        if general_ids[-1] < indic_ids[0] and indic_ids[-1] < special_ids[0]:
            P("  [PASS] Layout: General -> Indic -> Special (correct ordering)")
        else:
            P("  [WARN] Layout may not be properly ordered")

    # ─── CHECK 11: Added tokens detail ───
    P("\n" + "-"*60)
    P("CHECK 11: ALL ADDED (SPECIAL) TOKENS LISTING")
    P("-"*60)

    # Group by category
    categories = defaultdict(list)
    for t in added:
        c = t["content"]
        if "reserved_" in c:
            categories["reserved"].append(t)
        elif "lang:" in c:
            categories["language_tags"].append(t)
        elif "source:" in c:
            categories["source_tags"].append(t)
        elif "fim_" in c:
            categories["fim"].append(t)
        elif "vision" in c or "image" in c or "video" in c:
            categories["vision"].append(t)
        elif "think" in c or "reason" in c or "reflect" in c or "plan" in c.lower():
            categories["thinking"].append(t)
        elif "tool" in c or "function" in c or "api" in c:
            categories["tool_calling"].append(t)
        elif "code" in c or "output" in c or "error" in c or "stdin" in c or "stdout" in c or "stderr" in c or "file" in c:
            categories["code"].append(t)
        elif any(x in c for x in ["system","user","assistant","turn","context","instruction","response"]):
            categories["chat_roles"].append(t)
        elif any(x in c for x in ["begin_of_text","end_of_text","pad","unk","sep","cls","mask","newline","paragraph","document"]):
            categories["document_control"].append(t)
        elif "box" in c or "quad" in c or "object" in c:
            categories["vision_grounding"].append(t)
        elif c in [make_special("im_start"), make_special("im_end")]:
            categories["chat_format"].append(t)
        elif c == make_special("EOT"):
            categories["end_of_turn"].append(t)
        elif any(x in c for x in ["json","schema","arguments"]):
            categories["json_structure"].append(t)
        else:
            categories["other"].append(t)

    for cat, tokens in sorted(categories.items()):
        if cat == "reserved":
            P(f"\n  {cat} ({len(tokens)} tokens): IDs {tokens[0]['id']}-{tokens[-1]['id']}")
        else:
            P(f"\n  {cat} ({len(tokens)} tokens):")
            for t in tokens:
                P(f"    ID {t['id']:6d}: {t['content']}")

    # ─── CHECK 12: High-ID vocab sample (near end) ───
    P("\n" + "-"*60)
    P("CHECK 12: SAMPLE TOKENS FROM DIFFERENT ID RANGES")
    P("-"*60)

    # Sort vocab by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    non_special_sorted = [(k,v) for k,v in sorted_vocab if k not in added_set]

    ranges = [
        ("First 20 tokens (byte level)", 0, 20),
        ("Tokens 250-270 (early merges)", 250, 270),
        ("Tokens 1000-1020", 1000, 1020),
        ("Tokens 5000-5020", 5000, 5020),
        ("Tokens 50000-50020", 50000, 50020),
        ("Tokens 100000-100020", 100000, 100020),
    ]

    for label, start, end in ranges:
        P(f"\n  {label}:")
        slice_tokens = non_special_sorted[start:end]
        for t, tid in slice_tokens:
            d = dec(t)
            P(f"    ID {tid:6d}: {repr(d)[:60]}")

    # Last 30 non-special tokens (should be Indic)
    P("\n  Last 30 non-special tokens (should be Indic):")
    for t, tid in non_special_sorted[-30:]:
        d = dec(t)
        sc = indic_scripts(t)
        script_label = ", ".join(sc) if sc else "non-Indic"
        P(f"    ID {tid:6d}: {repr(d)[:40]:42s} [{script_label}]")

    # ─── CHECK 13: Token count summary breakdown ───
    P("\n" + "-"*60)
    P("CHECK 13: TOKEN COUNT BREAKDOWN")
    P("-"*60)
    
    # Count categories
    base_byte_tokens = 0
    for t in vocab:
        if t in added_set: continue
        if len(t) == 1:
            base_byte_tokens += 1
    
    P(f"  Single-char (byte) tokens: {base_byte_tokens}")
    P(f"  Non-special vocab tokens: {len(non_special_ids):,}")
    P(f"  Indic tokens: {len(indic_ids):,}")
    P(f"  General tokens: {len(general_ids):,}")
    P(f"  Special tokens: {len(special_ids)}")
    P(f"    - Base special: {len([t for t in added if 'reserved_' not in t['content']])}")
    P(f"    - Reserved: {reserved_count}")
    P(f"  TOTAL: {len(vocab):,}")

    # ─── FINAL SUMMARY ───
    P("\n" + "=" * 80)
    P("FINAL AUDIT SUMMARY")
    P("=" * 80)
    
    checks_text = "\n".join(out)
    pass_count = checks_text.count("[PASS]")
    fail_count = checks_text.count("[FAIL]") 
    warn_count = checks_text.count("[WARN]")
    
    P(f"  PASS: {pass_count}")
    P(f"  FAIL: {fail_count}")
    P(f"  WARN: {warn_count}")

    # Write report
    with open("tsai_131k_tokenizer/AUDIT_REPORT.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\n  Report saved to: tsai_131k_tokenizer/AUDIT_REPORT.txt")

# These are used in CHECK 13 just for counting
BASE_SPECIAL_EXPECTED = [
    "begin_of_text","end_of_text","pad","unk","sep","cls","mask","newline","paragraph","document",
    "system","user","assistant","tool","function","context","instruction","response","turn","end_turn",
    "code_begin","code_end","output_begin","output_end","error","stdin","stdout","stderr","file_begin","file_end",
]
ADDITIONAL_EXPECTED = [
    "fim_prefix","fim_middle","fim_suffix","fim_pad",
    "vision_start","vision_end","vision_pad","image_pad","video_pad",
]

if __name__ == "__main__":
    main()
