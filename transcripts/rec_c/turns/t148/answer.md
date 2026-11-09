**Submitted: no weight edits.** `/work/edited` is empty, and restoration of the original weights was verified. I did not find a validated recovery improvement within the 50-pass budget.

The strongest mechanistic finding was a controllable late visual-to-gripper pathway:
- Late VLM features distinguished empty closures from contact-associated states.
- A targeted rank-one edit to the final VLM value projection routed that feature through expert head 1 toward reopening.
- On reserved observations, reopening increased from **5% to 15% of empty-grasp action tokens** between half and full dose, without changing contact-state gripper signs.

However, this causal action change **did not improve recovery**:

| Fresh paired evaluation | Original | Edited |
|---|---:|---:|
| Successes, 352 episodes | 213 | 208 |
| Re-grasp-loop failures | 115 | 121 |

The success difference was **−1.42 ± 1.78 percentage points** (paired SE). The final 256-episode confirmation subset also regressed: **149 → 143 successes**. Earlier lift-only and attention interventions likewise failed confirmation.

These results establish control over a relevant pathway, **not that it is the cause of the loops or a successful repair**. Retaining the edit would contradict the validation criterion. Reverting preserves ordinary-task performance exactly, but I cannot claim improved recovery.

The experimental record and final decision are saved in `WORK/notes.md` and `WORK/final_report.json`.