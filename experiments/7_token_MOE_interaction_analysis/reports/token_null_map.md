# Token Null Routing Report

_Auto-generated on 2026-02-21T13:23:57.099307 UTC_

## 1. Top Tokens by Null Routing Score (Frequency-Gated)

| Rank | Token | Pg | Max Affinity | Null Routing Score |
|------|-------|----|--------------|--------------------|
| 1 | `,` | 3.14e-02 | -2.22e-16 | 3.14e+22 |
| 2 | `Ġthe` | 3.01e-02 | -1.11e-16 | 3.01e+22 |
| 3 | `.` | 2.19e-02 | -1.11e-16 | 2.19e+22 |
| 4 | `Ġ` | 1.59e-02 | -2.22e-16 | 1.59e+22 |
| 5 | `Ġof` | 1.57e-02 | -2.22e-16 | 1.57e+22 |
| 6 | `Ġto` | 1.56e-02 | -1.11e-16 | 1.56e+22 |
| 7 | `Ġand` | 1.52e-02 | -1.11e-16 | 1.52e+22 |
| 8 | `Ġa` | 1.29e-02 | -1.11e-16 | 1.29e+22 |
| 9 | `Ġin` | 1.05e-02 | -1.11e-16 | 1.05e+22 |
| 10 | `Ġis` | 7.78e-03 | -1.11e-16 | 7.78e+21 |
| 11 | `.Ċ` | 7.55e-03 | -1.11e-16 | 7.55e+21 |
| 12 | `Ġfor` | 5.98e-03 | -1.11e-16 | 5.97e+21 |
| 13 | `Ġthat` | 5.74e-03 | -1.11e-16 | 5.74e+21 |
| 14 | `Ċ` | 5.49e-03 | -1.11e-16 | 5.49e+21 |
| 15 | `:` | 4.79e-03 | -2.22e-16 | 4.79e+21 |
| 16 | `Ġ(` | 4.42e-03 | -2.22e-16 | 4.42e+21 |
| 17 | `Ġwith` | 4.39e-03 | -2.22e-16 | 4.39e+21 |
| 18 | `1` | 4.31e-03 | -2.22e-16 | 4.31e+21 |
| 19 | `Ġon` | 4.30e-03 | -2.22e-16 | 4.30e+21 |
| 20 | `ĠĠĠ` | 3.89e-03 | -1.11e-16 | 3.89e+21 |

## 2. Junk Token Candidates (Rare + No Affinity)

| Token | Pg | Max Affinity |
|-------|----|--------------|
| `À` | 9.99e-09 | -1.11e-16 |
| `Á` | 9.99e-09 | -1.11e-16 |
| `õ` | 9.99e-09 | -1.11e-16 |
| `ö` | 9.99e-09 | -1.11e-16 |
| `÷` | 9.99e-09 | -1.11e-16 |
| `ø` | 9.99e-09 | -1.11e-16 |
| `ù` | 9.99e-09 | -1.11e-16 |
| `ú` | 9.99e-09 | -1.11e-16 |
| `û` | 9.99e-09 | -1.11e-16 |
| `ü` | 9.99e-09 | -1.11e-16 |
| `ý` | 9.99e-09 | -1.11e-16 |
| `þ` | 9.99e-09 | -1.11e-16 |
| `ÿ` | 9.99e-09 | -1.11e-16 |
| `à±` | 9.99e-09 | -1.11e-16 |
| `âĢĭáŀ` | 9.99e-09 | -1.11e-16 |
| `Ġawá»įn` | 9.99e-09 | -1.11e-16 |
| `Ġgebru` | 9.99e-09 | -1.11e-16 |
| `Ġpesso` | 9.99e-09 | -1.11e-16 |
| `Æ°á»£` | 9.99e-09 | -1.11e-16 |
| `Æ°á»Ŀ` | 9.99e-09 | -1.11e-16 |

## 3. Null-Attracting Token Clusters

### Morphological fragments

- `Ġis` (Pg=7.8e-03, null_score=7.8e+21)

### Lexical glue tokens

- `Ġthe` (Pg=3.0e-02, null_score=3.0e+22)
- `Ġ` (Pg=1.6e-02, null_score=1.6e+22)
- `Ġof` (Pg=1.6e-02, null_score=1.6e+22)
- `Ġto` (Pg=1.6e-02, null_score=1.6e+22)
- `Ġand` (Pg=1.5e-02, null_score=1.5e+22)
- `Ġa` (Pg=1.3e-02, null_score=1.3e+22)
- `Ġin` (Pg=1.0e-02, null_score=1.0e+22)
- `Ġfor` (Pg=6.0e-03, null_score=6.0e+21)
- `Ġthat` (Pg=5.7e-03, null_score=5.7e+21)
- `Ġ(` (Pg=4.4e-03, null_score=4.4e+21)
- `Ġwith` (Pg=4.4e-03, null_score=4.4e+21)
- `Ġon` (Pg=4.3e-03, null_score=4.3e+21)
- `ĠĠĠ` (Pg=3.9e-03, null_score=3.9e+21)

### Structural separators

- `,` (Pg=3.1e-02, null_score=3.1e+22)
- `.` (Pg=2.2e-02, null_score=2.2e+22)

## 4. Notes

- Null routing score already includes frequency gating.
- Junk tokens are intentionally excluded from rankings.
- High scores indicate load on the null / shared pathway.
- Results stabilize as dataset size increases.
