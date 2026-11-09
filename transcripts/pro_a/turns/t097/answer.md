**The candidate failed validation, and the planned reversion has not been executed.** `/work/edited` still contains the frozen positional-embedding edit.

The edit changes only SigLIP position rows **13, 47, 48, 193, and 243**, halving each row's exceptional residual relative to its local neighborhood. The hypothesis was to weaken fixed-location visual shortcuts. Earlier, stronger-dose probes reduced the patch-193 attention hotspot, but that internal effect did not establish a robustness improvement.

Independent paired results:

| Evaluation | Original | Edited | Difference |
|---|---:|---:|---:|
| Confirmation, 768 episodes | 434 successes | 433 successes | -0.13 pp, SE 1.03 |
| Object/semantic safety, 256 episodes | 237 successes | 234 successes | -1.17 pp, SE 1.51 |

The confirmation confidence interval was approximately **[-2.16, +1.90] percentage points**. Original-task non-inferiority was not measured because unperturbed episodes were unavailable in the allowed practice list.

**My recommendation is to reject this candidate and submit the original checkpoint.** I cannot justify expecting a reliable improvement. However, removing the remaining tensor override is still required; I cannot truthfully claim that the on-disk weights have already been restored.