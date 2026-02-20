# Test 1 Comparison (Expected vs Got)

## Expected
- Winner selected from 100-step comparative run under fixed controls.
- No crash, no NaN/inf.
- Deterministic setup (same seed and training settings).

## Got
- lead_wo_rev final loss (step 100): TODO
- diff_rec final loss (step 100): TODO
- winner: TODO

## Decision Rationale
- Primary: lower NTP loss (`loss`)
- Secondary: stability and throughput consistency

## Follow-up
- Promote winner to Test 2 (20-step smoke with strict deterministic replay gate).
