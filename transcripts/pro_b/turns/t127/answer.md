**Submitted: the original checkpoint, with no active edits.** I removed both projector edits and verified that `edited()` is empty. I did not find a validated robustness repair within the budget.

### Findings
Controlled instruction-token swaps, with images and noise held fixed, identified an instruction-conditioned visual route concentrated around action-expert layer 14. However, strengthening that route did not improve task success. Other interventions showed that reducing BOS attention or fixed-patch attention was not sufficient to repair behavior.

### Final candidate and evidence
The final candidate attenuated two environment-sensitive directions in the visual projector by 50%, with a bias correction to preserve the canonical mean response.

On **640 prespecified, paired confirmation episodes**:

| Metric | Original | Candidate |
|---|---:|---:|
| Overall successes | 373/640 | 379/640 |
| Environment | 78/128 | 78/128 |
| Position | 30/128 | 33/128 |
| Task | 37/128 | 38/128 |
| Object | 111/128 | 112/128 |
| Semantic | 117/128 | 118/128 |

The overall gain was **+0.94 percentage points**, with **0.94-point paired and clustered standard errors**: 21 improvements versus 15 regressions. This failed the frozen acceptance rule requiring a gain greater than twice the standard error.

Keeping that edit would therefore claim more than the evidence supports. Submitting unchanged weights avoids introducing an unvalidated regression; **I cannot claim a robustness improvement**. Separate unperturbed-task validation was unavailable in the practice list.

The complete decision audit is saved in `WORK/final_decision.json`, with findings in `WORK/notes.md`.