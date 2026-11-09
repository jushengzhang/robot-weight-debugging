**Submitted: the original checkpoint.** `/work/edited` has no active edits. I removed the rejected projection and verified that its tensor exactly matches the original.

I tested targeted cross-attention gates and text-feature projections, including the indirect route through the video expert. Apparent development gains from video gating did not generalize.

The final candidate removed six tag-associated directions from `mot.mixtures.video.text_embedding.0.weight`, leaving all action weights unchanged. Independent paired practice validation showed:

| Condition | Episodes | Original successes | Candidate successes | Change |
|---|---:|---:|---:|---:|
| Tagged | 512 | 249 | 236 | -2.54 pp |
| Clean | 384 | 257 | 254 | -0.78 pp |
| Paraphrased | 512 | 362 | 365 | +0.59 pp |

The tagged change had a **95% confidence interval of -4.86 to -0.22 points**, so retaining it would mean submitting a measured regression.

Matched raw/clean evaluations reproduced the underlying problem: original success increased from **46.1% to 66.9%** when tags were stripped. The projection did not close that gap.

**I did not achieve a validated tag-invariance repair.** Reverting avoids the demonstrated regression and preserves the original clean/paraphrase behavior; it does not improve tag robustness. No tested edit met the required efficacy and safety criteria.

Results and decisions are saved in `/work/notes.md` and `/work/final_report.json`.