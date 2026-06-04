"""
Byte-length analysis for FINAL_TOKENIZER (GPT-2 style byte-level BPE, 131072 vocab).

Produces:
  tokens.csv     — one row per token_id
  coverage.csv   — POS_DIM coverage table
  REPORT.md      — summary report
  histogram.png  — byte-length histogram (if matplotlib available)
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "tokenizer"
DEFAULT_OUT = REPO_ROOT / "tokenizer" / "byte_analysis"

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
args = parser.parse_args()

TOKENIZER_DIR = args.tokenizer_dir
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# GPT-2 byte <-> unicode mapping (the canonical one from gpt2/tokenizers/ByteLevel).
# ---------------------------------------------------------------------------
def _bytes_to_unicode():
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
    return dict(zip(bs, [chr(c) for c in cs]))


_B2U = _bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}  # unicode char -> raw byte (0..255)


def piece_to_bytes(piece: str) -> bytes:
    """Decode a GPT-2 byte-level piece back to its raw byte sequence."""
    out = bytearray()
    for ch in piece:
        if ch in _U2B:
            out.append(_U2B[ch])
        else:
            # Shouldn't happen for normal pieces; encode as UTF-8 as a safety net.
            out.extend(ch.encode("utf-8", errors="replace"))
    return bytes(out)


# ---------------------------------------------------------------------------
# Load tokenizer + identify categories
# ---------------------------------------------------------------------------
print(f"Loading tokenizer from {TOKENIZER_PATH} ...", flush=True)
tok = Tokenizer.from_file(str(TOKENIZER_PATH))
with open(TOKENIZER_PATH) as f:
    raw = json.load(f)

added = raw.get("added_tokens", [])
added_by_id = {a["id"]: a for a in added}
special_ids = {a["id"] for a in added if a.get("special", False)}

vocab_size = tok.get_vocab_size(with_added_tokens=True)
print(f"vocab_size = {vocab_size}", flush=True)
print(f"added_tokens = {len(added)} ({len(special_ids)} marked special)", flush=True)

BYTEFB_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
SPECIAL_LOOK_RE = re.compile(r"^(<\|.*\|>|<[A-Za-z_/!][^>]*>)$")


def safe_repr(s, n=30):
    r = repr(s)
    if len(r) > n:
        r = r[: n - 1] + "…"
    return r


# ---------------------------------------------------------------------------
# Walk every token id
# ---------------------------------------------------------------------------
rows = []  # token_id, num_bytes, num_codepoints, category, piece_repr, piece_raw
missing = 0
for tid in range(vocab_size):
    piece = tok.id_to_token(tid)
    if piece is None:
        missing += 1
        rows.append((tid, 0, 0, "other", "<MISSING>", ""))
        continue

    # Categorize
    if BYTEFB_RE.match(piece):
        category = "bytefallback"
        byte_val = int(BYTEFB_RE.match(piece).group(1), 16)
        surface_bytes = bytes([byte_val])
        num_bytes = 1
        num_codepoints = 1
    elif tid in special_ids or (tid in added_by_id and SPECIAL_LOOK_RE.match(piece)):
        category = "special"
        surface_bytes = piece.encode("utf-8")
        num_bytes = len(surface_bytes)
        num_codepoints = len(piece)
    else:
        category = "normal"
        surface_bytes = piece_to_bytes(piece)
        num_bytes = len(surface_bytes)
        try:
            num_codepoints = len(surface_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            # Partial UTF-8 fragment (common for BPE merges that split a codepoint).
            num_codepoints = -1  # sentinel: undecodable as standalone

    rows.append((tid, num_bytes, num_codepoints, category, safe_repr(piece), piece))

if missing:
    print(f"WARNING: {missing} token ids returned None from id_to_token", flush=True)


# ---------------------------------------------------------------------------
# Write tokens.csv
# ---------------------------------------------------------------------------
tokens_csv = OUT / "tokens.csv"
with open(tokens_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["token_id", "num_bytes", "num_codepoints", "category", "piece_repr"])
    for tid, nb, nc, cat, prepr, _ in rows:
        w.writerow([tid, nb, nc, cat, prepr])
print(f"Wrote {tokens_csv} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# coverage.csv
# ---------------------------------------------------------------------------
POS_DIMS = [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64]
n_all = len(rows)
n_excl = sum(1 for r in rows if r[3] in ("normal", "bytefallback"))

cov_rows = []
for pd in POS_DIMS:
    n_fit_all = sum(1 for r in rows if r[1] <= pd)
    n_fit_excl = sum(
        1 for r in rows if r[1] <= pd and r[3] in ("normal", "bytefallback")
    )
    cov_rows.append(
        (
            pd,
            n_fit_excl,
            100.0 * n_fit_excl / max(n_excl, 1),
            n_fit_all,
            100.0 * n_fit_all / max(n_all, 1),
        )
    )

coverage_csv = OUT / "coverage.csv"
with open(coverage_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "POS_DIM",
            "num_fit_excl_special",
            "pct_fit_excl_special",
            "num_fit_all",
            "pct_fit_all",
        ]
    )
    for pd, ne, pe, na, pa in cov_rows:
        w.writerow([pd, ne, f"{pe:.4f}", na, f"{pa:.4f}"])
print(f"Wrote {coverage_csv}", flush=True)


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------
cat_counts = Counter(r[3] for r in rows)
total_bytes = sum(r[1] for r in rows)

# Histogram bins
BINS = [
    ("1 byte", lambda n: n == 1),
    ("2 bytes", lambda n: n == 2),
    ("3 bytes", lambda n: n == 3),
    ("4 bytes", lambda n: n == 4),
    ("5-8 bytes", lambda n: 5 <= n <= 8),
    ("9-12 bytes", lambda n: 9 <= n <= 12),
    ("13-16 bytes", lambda n: 13 <= n <= 16),
    ("17-24 bytes", lambda n: 17 <= n <= 24),
    ("25-32 bytes", lambda n: 25 <= n <= 32),
    ("33-48 bytes", lambda n: 33 <= n <= 48),
    ("49+ bytes", lambda n: n >= 49),
]
bin_counts = [(label, sum(1 for r in rows if pred(r[1]))) for label, pred in BINS]

# longest 20 overall
sorted_by_len = sorted(rows, key=lambda r: (-r[1], r[0]))
top20 = sorted_by_len[:20]

# tokens truncated at POS_DIM=32 (excl. special, since the paper is about normal+bytefallback)
trunc32_excl = [r for r in rows if r[3] in ("normal", "bytefallback") and r[1] > 32]
trunc32_excl_sorted = sorted(trunc32_excl, key=lambda r: -r[1])
trunc32_all = [r for r in rows if r[1] > 32]


def category_stats(cat):
    vals = [r[1] for r in rows if r[3] == cat]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    p99_idx = max(0, int(round(0.99 * (n - 1))))
    return {
        "count": n,
        "mean": sum(vals) / n,
        "max": max(vals),
        "p99": vals_sorted[p99_idx],
    }


cat_stats = {
    c: category_stats(c) for c in ["normal", "bytefallback", "special", "other"]
}


# Sanity check: re-tokenize surface form for 10 random normal tokens
random.seed(0)
normal_rows = [r for r in rows if r[3] == "normal" and r[1] > 0 and r[1] < 30]
sample_rows = random.sample(normal_rows, min(10, len(normal_rows)))
sanity = []
for tid, nb, nc, cat, prepr, piece in sample_rows:
    try:
        surface_bytes = piece_to_bytes(piece)
        surface = surface_bytes.decode("utf-8", errors="replace")
        enc = tok.encode(surface, add_special_tokens=False)
        ids = enc.ids
        ok = ids == [tid]
        sanity.append((tid, prepr, ids, ok))
    except Exception as e:
        sanity.append((tid, prepr, f"ERR:{e}", False))


def fmt_pct(num, denom):
    return f"{100.0 * num / max(denom, 1):.2f}%"


lines = []
lines.append("# Tokenizer Byte-Length Analysis — FINAL_TOKENIZER\n")
lines.append(f"- **Tokenizer path:** `{TOKENIZER_PATH}`")
lines.append("- **Type:** GPT-2 byte-level BPE (HF `tokenizers` format)")
lines.append(f"- **Vocab size:** {vocab_size}")
lines.append(f"- **Total bytes (sum of surface-form lengths):** {total_bytes:,}")
lines.append("")
lines.append("## Category counts")
lines.append("")
lines.append("| Category | Count |")
lines.append("|---|---:|")
for c in ["normal", "bytefallback", "special", "other"]:
    lines.append(f"| {c} | {cat_counts.get(c, 0)} |")
lines.append("")

lines.append("## Byte-length distribution")
lines.append("")
lines.append("| byte_count_bin | num_tokens | pct |")
lines.append("|---|---:|---:|")
for label, n in bin_counts:
    lines.append(f"| {label} | {n} | {fmt_pct(n, len(rows))} |")
lines.append("")

lines.append("## Coverage at common POS_DIM values")
lines.append("")
lines.append(
    "| POS_DIM | fit (normal+bytefallback) | pct | fit (all incl. special) | pct |"
)
lines.append("|---:|---:|---:|---:|---:|")
for pd, ne, pe, na, pa in cov_rows:
    lines.append(f"| {pd} | {ne} | {pe:.2f}% | {na} | {pa:.2f}% |")
lines.append("")


# Key takeaways
def cov_for(pd):
    for row in cov_rows:
        if row[0] == pd:
            return row
    return None


c16 = cov_for(16)
c32 = cov_for(32)
lines.append(
    f"- **POS_DIM=16** covers **{c16[2]:.2f}%** of normal+bytefallback tokens "
    f"({c16[4]:.2f}% of all)."
)
lines.append(
    f"- **POS_DIM=32** covers **{c32[2]:.2f}%** of normal+bytefallback tokens "
    f"({c32[4]:.2f}% of all)."
)
lines.append(
    f"- **POS_DIM=32 truncates {len(trunc32_excl)} normal+bytefallback tokens** "
    f"(and {len(trunc32_all)} tokens overall, including specials)."
)
lines.append("")
lines.append("### Longest 20 normal+bytefallback tokens truncated at POS_DIM=32")
lines.append("")
lines.append("| token_id | num_bytes | category | piece_repr |")
lines.append("|---:|---:|---|---|")
for r in trunc32_excl_sorted[:20]:
    lines.append(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]} |")
if not trunc32_excl_sorted:
    lines.append("| _(none)_ | | | |")
lines.append("")

lines.append("## Top 20 longest tokens overall")
lines.append("")
lines.append("| token_id | num_bytes | num_codepoints | category | piece_repr |")
lines.append("|---:|---:|---:|---|---|")
for r in top20:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
lines.append("")

lines.append("## Category breakdown")
lines.append("")
lines.append("| category | count | mean num_bytes | max num_bytes | p99 num_bytes |")
lines.append("|---|---:|---:|---:|---:|")
for c, s in cat_stats.items():
    if s is None:
        lines.append(f"| {c} | 0 | — | — | — |")
    else:
        lines.append(
            f"| {c} | {s['count']} | {s['mean']:.2f} | {s['max']} | {s['p99']} |"
        )
lines.append("")

lines.append("## Surface-form recovery sanity check (10 random normal tokens)")
lines.append("")
lines.append("| token_id | piece_repr | re-encoded ids | pass |")
lines.append("|---:|---|---|:---:|")
n_pass = 0
for tid, prepr, ids, ok in sanity:
    n_pass += int(bool(ok))
    lines.append(f"| {tid} | {prepr} | {ids} | {'✓' if ok else '✗'} |")
lines.append("")
lines.append(f"**{n_pass}/{len(sanity)} round-trip correctly.**")
lines.append("")

# Largest truncated token at POS_DIM=32
if trunc32_excl_sorted:
    largest_trunc32 = trunc32_excl_sorted[0][1]
else:
    largest_trunc32 = 0
lines.append("---")
if largest_trunc32:
    tail = f"The largest token that gets truncated at POS_DIM=32 is {largest_trunc32} bytes."
else:
    longest_byte_len = max(r[1] for r in rows if r[3] in ("normal", "bytefallback"))
    tail = (
        "No normal+bytefallback token exceeds 32 bytes — the longest such token in the vocabulary "
        f"is exactly {longest_byte_len} bytes, so POS_DIM=32 truncates nothing."
    )
lines.append(
    f"**One-liner:** POS_DIM=32 covers {c32[2]:.2f}% of normal+bytefallback tokens, "
    f"{c32[4]:.2f}% of all tokens (including specials). {tail}"
)

report_md = OUT / "REPORT.md"
report_md.write_text("\n".join(lines))
print(f"Wrote {report_md}", flush=True)


# ---------------------------------------------------------------------------
# histogram.png (optional)
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bytes_arr = [r[1] for r in rows]
    max_b = max(bytes_arr)
    bins = list(range(0, min(max_b, 80) + 2))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [min(b, 80) for b in bytes_arr],
        bins=bins,
        color="#4477aa",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("token surface-form length (bytes)")
    ax.set_ylabel("number of tokens (log scale)")
    ax.set_title(f"FINAL_TOKENIZER byte-length distribution (vocab={vocab_size})")
    ax.axvline(16, color="#ee6677", linestyle="--", linewidth=1.2)
    ax.axvline(32, color="#228833", linestyle="--", linewidth=1.2)
    ax.text(
        16.5,
        ax.get_ylim()[1] * 0.4,
        f"POS_DIM=16\n{c16[2]:.2f}%",
        color="#ee6677",
        fontsize=9,
    )
    ax.text(
        32.5,
        ax.get_ylim()[1] * 0.4,
        f"POS_DIM=32\n{c32[2]:.2f}%",
        color="#228833",
        fontsize=9,
    )
    fig.tight_layout()
    png = OUT / "histogram.png"
    fig.savefig(png, dpi=160)
    print(f"Wrote {png}", flush=True)
except Exception as e:
    print(f"Skipping histogram.png: {e}", flush=True)


# ---------------------------------------------------------------------------
# Final one-line summary
# ---------------------------------------------------------------------------
print("")
print(
    f"POS_DIM=32 covers {c32[2]:.2f}% of normal+bytefallback tokens, "
    f"{c32[4]:.2f}% of all tokens (including specials). {tail}"
)
