**No validated repair was found. The 30 candidate edits remain in `/work/edited`; I did not execute the planned rollback before the final pass.**

The candidate applies a half-strength, rank-115 contrast-subspace correction to the video expert’s cross-attention value weights. Queries, keys, biases, and the action expert remain unchanged. Its intended mechanism was to suppress tag-associated text values before they contaminate the observation features read by the action expert.

Independent paired validation:

| Condition | Original | Candidate | Difference |
|---|---:|---:|---:|
| Tagged instructions | 229/512 | 230/512 | +0.20 pp |
| Clean instructions | 343/512 | 346/512 | +0.59 pp |
| Language paraphrases | 268/384 | 271/384 | +0.78 pp |

The tagged effect has **SE 1.12 pp and 95% CI [-2.01, +2.40] pp**. It does not meet the required gain criterion.

The diagnostic interventions implicated the video text-conditioning route, and value-side corrections improved action agreement more than matched key-side corrections. However, that probe improvement did **not** translate into validated rollout gains: the clean-versus-tagged success gap remained about 22 points.

Under the stated retention rule, this candidate should be rejected. I cannot honestly claim that the active submission makes FastWAM ignore variant tags.