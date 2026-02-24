"""Deep audit #2: Find anything else that might violate supervisor requirements.

Checks beyond the first audit:
1. Consistency between tokenizer.json, tokenizer_config.json, and special_tokens_map.json
2. Non-English/non-Indic words that slipped through the blocked-script filter
   (e.g. French, German, Spanish subwords that use Latin chars but are clearly foreign)
3. added_tokens_decoder in config matches added_tokens in tokenizer.json
4. Duplicate or near-duplicate tokens
5. Tokens that are pure whitespace > 16 chars (excessive indentation)
6. build_clean_tokenizer.py consistency with output
7. Any token ID gaps or overlaps
"""

import json
from collections import Counter, defaultdict

def make_tok(name):
    return '<' + '|' + name + '|' + '>'

def main():
    print("=" * 70)
    print("DEEP AUDIT #2: Finding remaining issues")
    print("=" * 70)

    # Load all three files
    with open('../tsai_131k_tokenizer/tokenizer.json', 'r', encoding='utf-8') as f:
        tok = json.load(f)
    with open('../tsai_131k_tokenizer/tokenizer_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open('../tsai_131k_tokenizer/special_tokens_map.json', 'r', encoding='utf-8') as f:
        smap = json.load(f)

    vocab = tok['model']['vocab']
    added = tok.get('added_tokens', [])
    added_set = {t['content'] for t in added}
    
    issues = []

    # ─── CHECK A: Cross-file consistency ───
    print("\n[A] CROSS-FILE CONSISTENCY")
    print("-" * 50)
    
    # Check bos/eos/pad/unk in config match vocab
    for field in ['bos_token', 'eos_token', 'pad_token']:
        val = config.get(field)
        if val and val not in vocab:
            issues.append(f"tokenizer_config.json '{field}' = '{val}' NOT in vocab!")
            print(f"  [ISSUE] {field} in config = '{val}' NOT IN VOCAB")
        elif val:
            print(f"  [OK] config.{field} = '{val}' (ID {vocab[val]})")

    # Check special_tokens_map matches
    for field in ['bos_token', 'eos_token', 'pad_token', 'unk_token']:
        sval = smap.get(field)
        cval = config.get(field)
        if sval and cval and sval != cval:
            issues.append(f"Mismatch: special_tokens_map.{field}='{sval}' vs config.{field}='{cval}'")
            print(f"  [ISSUE] {field} mismatch: map='{sval}' vs config='{cval}'")
        elif sval and cval:
            print(f"  [OK] {field} consistent across files: '{sval}'")
        elif sval and not cval:
            issues.append(f"config missing {field} (map has '{sval}')")
            print(f"  [ISSUE] config missing {field} (map has '{sval}')")
        elif cval and not sval:
            issues.append(f"special_tokens_map missing {field} (config has '{cval}')")
            print(f"  [WARN] special_tokens_map missing {field} (config has '{cval}')")

    # ─── CHECK B: added_tokens_decoder in config vs added_tokens in tokenizer ───
    print("\n[B] ADDED_TOKENS_DECODER CONSISTENCY")
    print("-" * 50)
    
    decoder = config.get('added_tokens_decoder', {})
    decoder_ids = set(int(k) for k in decoder.keys())
    tok_ids = set(t['id'] for t in added)
    
    missing_in_decoder = tok_ids - decoder_ids
    extra_in_decoder = decoder_ids - tok_ids
    
    if missing_in_decoder:
        print(f"  [ISSUE] {len(missing_in_decoder)} tokens in tokenizer.json but NOT in config decoder")
        for mid in sorted(missing_in_decoder)[:5]:
            t = [x for x in added if x['id'] == mid][0]
            print(f"    ID {mid}: {t['content']}")
        issues.append(f"{len(missing_in_decoder)} tokens missing from added_tokens_decoder")
    else:
        print(f"  [OK] All {len(tok_ids)} added_tokens present in decoder")
    
    if extra_in_decoder:
        print(f"  [ISSUE] {len(extra_in_decoder)} extra entries in decoder not in tokenizer.json")
        issues.append(f"{len(extra_in_decoder)} extra entries in added_tokens_decoder")
    
    # Check content matches
    mismatches = 0
    for t in added:
        tid = str(t['id'])
        if tid in decoder:
            if decoder[tid]['content'] != t['content']:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  [ISSUE] ID {tid}: tok='{t['content']}' vs decoder='{decoder[tid]['content']}'")
    if mismatches:
        print(f"  [ISSUE] {mismatches} content mismatches between files")
        issues.append(f"{mismatches} content mismatches in added_tokens_decoder")
    else:
        print("  [OK] All contents match")

    # ─── CHECK C: Token ID gaps/overlaps ───
    print("\n[C] TOKEN ID INTEGRITY")
    print("-" * 50)
    
    all_ids = sorted(vocab.values())
    expected_ids = list(range(131072))
    
    id_set = set(all_ids)
    missing_ids = set(expected_ids) - id_set
    duplicate_ids = [x for x, count in Counter(all_ids).items() if count > 1]
    
    if missing_ids:
        print(f"  [ISSUE] {len(missing_ids)} missing IDs in vocab")
        issues.append(f"{len(missing_ids)} missing token IDs")
    else:
        print(f"  [OK] All IDs 0-131071 present, no gaps")
    
    if duplicate_ids:
        print(f"  [ISSUE] {len(duplicate_ids)} duplicate IDs")
        issues.append(f"{len(duplicate_ids)} duplicate token IDs")
    else:
        print("  [OK] No duplicate IDs")

    # ─── CHECK D: Foreign-language Latin-script tokens ───
    print("\n[D] SUSPICIOUS FOREIGN-LANGUAGE TOKENS (Latin script but non-English)")
    print("-" * 50)
    
    # Common non-English Latin-script patterns that suggest French/German/Spanish/Portuguese/etc.
    foreign_markers = {
        'French': ['avoir', 'faire', 'peut', 'cette', 'sont', 'dans', 'pour', 'avec', 'tout',
                   'mais', 'elle', 'nous', 'vous', 'leur', 'comme', 'bien', 'fait', 'encore',
                   'autre', 'entre', 'depuis', 'aussi', 'quand', 'faire', 'apr', 'jusqu',
                   'peuvent', 'selon', 'toujours', 'parce', 'quelques', 'ailleurs',
                   'cependant', 'souvent', 'davantage', 'notamment', 'certains'],
        'German': ['nicht', 'auch', 'sich', 'noch', 'nach', 'dann', 'diese', 'werden',
                   'durch', 'wurde', 'hatte', 'sehr', 'sein', 'haben', 'schon', 'immer',
                   'zwischen', 'bereits', 'jedoch', 'sowie', 'dabei', 'gegen', 'unter',
                   'weil', 'heute', 'beiden', 'eigentlich', 'vielleicht', 'manchmal'],
        'Spanish': ['pero', 'como', 'para', 'sobre', 'este', 'tiene', 'puede', 'desde',
                    'hasta', 'todos', 'entre', 'otro', 'sido', 'cuando', 'donde', 'mismo',
                    'tambi', 'ning', 'entonces', 'mejor', 'siempre', 'antes', 'algunos'],
        'Dutch': ['niet', 'voor', 'maar', 'zijn', 'heeft', 'werd', 'naar', 'deze',
                  'door', 'heel', 'omdat', 'alleen', 'veel', 'weer', 'tegen', 'tijdens'],
        'Portuguese': ['como', 'para', 'isso', 'ainda', 'onde', 'muito', 'mesmo',
                       'sobre', 'depois', 'parte', 'quando', 'entre', 'outros', 'sendo'],
    }
    
    # Build GPT-2 decoder
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs = list(bs); n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    c2b = {chr(c): b for b, c in zip(bs, cs)}
    
    def dec(t):
        try: return bytes(c2b[c] for c in t).decode("utf-8", errors="replace")
        except KeyError: return t
    
    foreign_found = defaultdict(list)
    for token in vocab:
        if token in added_set:
            continue
        d = dec(token).strip().lower()
        if len(d) < 4:  # Skip short tokens
            continue
        for lang, words in foreign_markers.items():
            for w in words:
                if d == w or d == ' ' + w:
                    foreign_found[lang].append((token, vocab[token], d))
                    break
    
    total_foreign = sum(len(v) for v in foreign_found.values())
    if foreign_found:
        print(f"  [INFO] Found {total_foreign} tokens matching common foreign words")
        print("  (These use Latin script so they passed the Unicode filter, but are")
        print("   clearly non-English. They may still be useful for code comments,")
        print("   variable names, etc.)")
        for lang, tokens in sorted(foreign_found.items(), key=lambda x: -len(x[1])):
            print(f"\n  {lang} ({len(tokens)} tokens):")
            for t, tid, d in tokens[:10]:
                print(f"    ID {tid}: '{d}'")
    else:
        print("  [OK] No suspicious foreign-language tokens found")

    # ─── CHECK E: Excessive whitespace tokens ───
    print("\n[E] EXCESSIVE WHITESPACE TOKENS")
    print("-" * 50)
    
    whitespace_tokens = []
    for token in vocab:
        if token in added_set:
            continue
        d = dec(token)
        if d and all(c in ' \t' for c in d) and len(d) > 16:
            whitespace_tokens.append((token, vocab[token], len(d), repr(d)))
    
    if whitespace_tokens:
        print(f"  [INFO] {len(whitespace_tokens)} tokens are pure whitespace > 16 chars:")
        for t, tid, length, r in sorted(whitespace_tokens, key=lambda x: -x[2]):
            print(f"    ID {tid}: {length} chars of whitespace")
    else:
        print("  [OK] No excessive whitespace tokens")

    # ─── CHECK F: build_clean_tokenizer.py generates correct config ───
    print("\n[F] BUILD SCRIPT CONSISTENCY")
    print("-" * 50)
    
    # Check if build_clean_tokenizer.py would produce wrong tokens
    # It hardcodes the special_tokens_map - let's check
    try:
        with open('../build_clean_tokenizer.py', 'r', encoding='utf-8') as f:
            src = f.read()
        
        if 'end_of_text' in src and 'pad_token' in src:
            # Check if the build script still writes pad_token as end_of_text
            if '"pad_token": "<|end_of_text|>"' in src or "'pad_token': '<|end_of_text|>'" in src:
                issues.append("build_clean_tokenizer.py still writes pad_token as end_of_text (line ~411)")
                print("  [ISSUE] build_clean_tokenizer.py still sets pad_token = end_of_text")
                print("          If the tokenizer is rebuilt, it will overwrite our fix!")
                print("          Fix line ~411 in build_clean_tokenizer.py")
            else:
                print("  [OK] Build script pad_token looks correct")
        
        if 'unk_token' not in src:
            issues.append("build_clean_tokenizer.py does not write unk_token to special_tokens_map.json")
            print("  [ISSUE] build_clean_tokenizer.py does not add unk_token to special_tokens_map")
            print("          If rebuilt, unk_token will be missing again")
    except Exception as e:
        print(f"  [SKIP] Could not check build script: {e}")

    # ─── CHECK G: tokenizer_config.json missing unk_token field ───
    print("\n[G] CONFIG COMPLETENESS")
    print("-" * 50)
    
    expected_fields = ['bos_token', 'eos_token', 'pad_token', 'unk_token']
    for field in expected_fields:
        if field in config:
            print(f"  [OK] config has {field} = '{config[field]}'")
        else:
            issues.append(f"tokenizer_config.json missing '{field}'")
            print(f"  [ISSUE] config missing '{field}'")

    # ─── SUMMARY ───
    print("\n" + "=" * 70)
    print("DEEP AUDIT SUMMARY")
    print("=" * 70)
    
    if issues:
        print(f"\n  Found {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n  No issues found!")

    print()

if __name__ == '__main__':
    main()
