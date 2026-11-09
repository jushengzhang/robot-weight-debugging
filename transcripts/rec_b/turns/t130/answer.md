**Submitted: the original checkpoint, unchanged.** `/work/edited` is empty; all temporary edits were removed. I did not establish a validated recovery improvement.

### Evidence for reverting
The two strongest diagnostic candidates failed to generalize:

| Candidate | Diagnostic success | Pooled validation success | Rescues / regressions |
|---|---:|---:|---:|
| Gripper-specific amplification in expert L2h0 queries | 26/48 -> 30/48 | 133/254 -> 133/254 | 11 / 11 |
| Selective query softening in L7h2, L15h1, L16h6 | 26/48 -> 31/48 | 133/254 -> 133/254 | 12 / 12 |

Both validation gains were **0.0 percentage points**, with paired standard errors of 1.85 and 1.93 points, respectively. Neither met your retention criterion.

### What the investigation established
- Attention routing was causally modifiable: query softening increased image attention in one loop from approximately **36% to 66%**. However, it also substantially changed successful controls, and recovery gains did not generalize.
- Wrist fixation, positional-register attention, and gripper-aligned pathways were not demonstrated to be failure-specific causes.
- Looping episodes did not have a higher gripper-flip rate per step than successful episodes in the diagnostic audit. Large flip counts therefore partly reflected prolonged failure, rather than an isolated gripper-oscillation defect.

Pre-failure preservation checks showed no convincing aggregate damage, but these were not ordinary LIBERO evaluations.

**I do not expect unchanged weights to improve recovery.** Reverting avoids submitting an unsupported intervention and preserves baseline ordinary performance by construction. The findings and final paired statistics are saved in `notes.md` and `final_report.json`.