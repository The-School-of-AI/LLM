"""The Kronecker codec must see each token's raw bytes, not decode([id]) text.

decode() renders a partial UTF-8 character as U+FFFD, so in a byte-level BPE
vocabulary many distinct tokens decode to the same string and would share one
Kronecker embedding. These tests pin the fix: raw bytes give zero collisions.
"""

from collections import Counter
from pathlib import Path

import pytest

from lightninglm.data.token_bytes import check_no_collisions, token_bytes, vocab_token_bytes

TSAI_TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_reordered.json"


def _n_shared(strings):
    return sum(n for n in Counter(strings).values() if n > 1)


def test_gpt2_raw_bytes_have_no_collisions():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    vocab = vocab_token_bytes(enc, enc.n_vocab)
    assert _n_shared(vocab) == 0
    check_no_collisions(vocab)
    # The bug being fixed: decode([id]) ties hundreds of GPT-2 tokens together.
    assert _n_shared(enc.decode([i]) for i in range(enc.n_vocab)) > 300


def test_tsai_131k_raw_bytes_have_no_collisions():
    transformers = pytest.importorskip("transformers")
    if not TSAI_TOKENIZER.exists():
        pytest.skip("TSAI-131K tokenizer file not present")
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(TSAI_TOKENIZER))
    vocab = vocab_token_bytes(tok)
    assert len(vocab) == len(tok)
    assert _n_shared(vocab) == 0
    check_no_collisions(vocab)
    decoded = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tok))]
    assert _n_shared(decoded) > 1000


def test_byte_level_tokens_round_trip_through_their_own_bytes():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    for i in (0, 100, 5000, 50256):
        assert token_bytes(enc, i) == enc.decode_single_token_bytes(i)


def test_collision_check_rejects_shared_bytes():
    with pytest.raises(ValueError, match="share bytes"):
        check_no_collisions([b"a", b"\xef\xbf\xbd", b"\xef\xbf\xbd"])
