# Token Null Routing Report

_Auto-generated on 2026-02-08T09:52:50.116631 UTC_

## 1. Top Tokens by Null Routing Score (Frequency-Gated)

| Rank | Token | Pg | Max Affinity | Null Routing Score |
|------|-------|----|--------------|--------------------|
| 1 | `Ġand` | 2.32e-02 | 3.68e-03 | 1.23e+04 |
| 2 | `,` | 3.69e-02 | 7.84e-02 | 2.02e+03 |
| 3 | `Ġin` | 1.30e-02 | 6.41e-02 | 1.82e+03 |
| 4 | `Ġfrom` | 3.35e-03 | 3.67e-03 | 1.67e+03 |
| 5 | `Ġthe` | 3.63e-02 | 2.96e-01 | 4.15e+02 |
| 6 | `.` | 2.78e-02 | 2.51e-01 | 1.89e+02 |
| 7 | `Ġfor` | 8.56e-03 | 2.45e-02 | 1.54e+02 |
| 8 | `Ġon` | 5.78e-03 | 8.96e-03 | 1.47e+02 |
| 9 | `Ġwith` | 6.37e-03 | 6.17e-02 | 1.08e+02 |
| 10 | `Ġof` | 1.98e-02 | 2.12e-01 | 8.19e+01 |

## 2. Junk Token Candidates (Rare + No Affinity)

| Token | Pg | Max Affinity |
|-------|----|--------------|
| `{` | 9.98e-09 | 2.22e-16 |
| `À` | 9.98e-09 | 2.22e-16 |
| `Á` | 9.98e-09 | 2.22e-16 |
| `Õ` | 9.98e-09 | 2.22e-16 |
| `Ø` | 9.98e-09 | 2.22e-16 |
| `Þ` | 9.98e-09 | 2.22e-16 |
| `å` | 9.98e-09 | 2.22e-16 |
| `æ` | 9.98e-09 | 2.22e-16 |
| `ç` | 9.98e-09 | 2.22e-16 |
| `ð` | 9.98e-09 | 2.22e-16 |

## 3. Null-Attracting Token Clusters

### Morphological fragments


### Lexical glue tokens

- `Ġand` (Pg=2.3e-02, null_score=1.2e+04)
- `Ġin` (Pg=1.3e-02, null_score=1.8e+03)
- `Ġfrom` (Pg=3.3e-03, null_score=1.7e+03)
- `Ġthe` (Pg=3.6e-02, null_score=4.2e+02)
- `Ġfor` (Pg=8.6e-03, null_score=1.5e+02)
- `Ġon` (Pg=5.8e-03, null_score=1.5e+02)
- `Ġwith` (Pg=6.4e-03, null_score=1.1e+02)
- `Ġof` (Pg=2.0e-02, null_score=8.2e+01)

### Structural separators

- `,` (Pg=3.7e-02, null_score=2.0e+03)
- `.` (Pg=2.8e-02, null_score=1.9e+02)

## 4. Notes

- Null routing score already includes frequency gating.
- Junk tokens are intentionally excluded from rankings.
- High scores indicate load on the null / shared pathway.
- Results stabilize as dataset size increases.
