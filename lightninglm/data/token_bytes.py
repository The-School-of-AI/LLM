"""Raw bytes of each vocabulary token, for the Kronecker codec.

Do NOT build the codec's vocabulary with ``tokenizer.decode([id])``. In a
byte-level BPE vocabulary (GPT-2, TSAI-131K) many tokens hold only part of a
multi-byte UTF-8 character. ``decode`` cannot render a partial character, so it
returns U+FFFD instead, and every such token gets the same bytes and therefore
the same Kronecker embedding. On TSAI-131K that ties 1,151 tokens to shared
codes (657 of them to one); on GPT-2, 344 tokens collapse into 15 codes.

``token_bytes`` returns the exact bytes the tokenizer assigned to each id, so no
two tokens can collide (apart from truncation beyond the codec's POS_DIM).
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: printable stand-in char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tokenizer, token_id: int) -> bytes:
    """Exact bytes of one token. Works for tiktoken and HF tokenizers."""
    # tiktoken exposes the bytes directly.
    if hasattr(tokenizer, "decode_single_token_bytes"):
        return tokenizer.decode_single_token_bytes(token_id)

    tok = tokenizer.convert_ids_to_tokens(token_id)
    if tok is None:
        return b""
    if token_id in set(getattr(tokenizer, "all_special_ids", ()) or ()):
        return tok.encode("utf-8")

    # Byte-level BPE (GPT-2, TSAI-131K): each char stands for one byte.
    dec = _byte_decoder()
    if all(c in dec for c in tok):
        return bytes(dec[c] for c in tok)

    # SentencePiece-style vocabularies.
    m = _BYTE_FALLBACK.match(tok)
    if m:
        return bytes([int(m.group(1), 16)])
    return tok.replace("▁", " ").encode("utf-8")


def vocab_token_bytes(tokenizer, vocab_size: int | None = None) -> list[bytes]:
    """Bytes for every id, with the old ``<unk_i>`` placeholder for empty tokens."""
    n = len(tokenizer) if vocab_size is None else vocab_size
    out = []
    for i in range(n):
        try:
            b = token_bytes(tokenizer, i)
        except Exception:
            b = b""
        out.append(b if b else f"<unk_{i}>".encode("utf-8"))
    return out


def check_no_collisions(vocab: list[bytes], pos_dim: int = 32) -> None:
    """Raise if two tokens share bytes within the codec's first ``pos_dim`` bytes.

    Tokens longer than ``pos_dim`` that collide only after truncation are a
    separate, known limit of the codec and are reported, not raised.
    """
    full, cut = defaultdict(list), defaultdict(list)
    for i, b in enumerate(vocab):
        full[bytes(b)].append(i)
        cut[bytes(b[:pos_dim])].append(i)
    dup = {k: v for k, v in full.items() if len(v) > 1}
    if dup:
        k, v = next(iter(dup.items()))
        raise ValueError(
            f"{sum(len(v) for v in dup.values())} tokens share bytes with another "
            f"token (e.g. ids {v[:5]} -> {k!r}). Build the vocabulary with "
            "token_bytes(), not tokenizer.decode([id])."
        )
    trunc = sum(len(v) for v in cut.values() if len(v) > 1)
    if trunc:
        print(f"  note: {trunc} tokens collide only after truncation to {pos_dim} bytes")
