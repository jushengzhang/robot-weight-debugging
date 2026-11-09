**Submitted: the original checkpoint.** `/work/edited` is empty, verified after removing all 308 candidate edits. I did not find a validated, safe repair.

### What I tested
The frozen candidate combined:
- A rank-256 projection in both experts’ text-input weights, targeting tag-associated context changes.
- Selective cross-attention query/key gates intended to suppress suffix features.

Diagnostics implicated both action- and video-expert text pathways. Suppressing detected suffix features left substantial context differences, suggesting contamination beyond the suffix itself. The combined edit reduced diagnostic raw-to-clean action distance by about **35%**, but that improvement did not reliably translate into task success.

### Independent validation
The candidate was frozen before these paired practice evaluations; every planned batch was included.

| Condition | Episodes | Original | Candidate | Change, paired SE |
|---|---:|---:|---:|---:|
| Tagged | 1,024 | 49.4% | 50.4% | +0.98 ± 1.21 pp |
| Clean | 768 | 68.8% | 66.1% | −2.60 ± 1.35 pp |
| Language paraphrases | 640 | 68.8% | 67.2% | −1.56 ± 1.41 pp |

The tagged gain’s 95% confidence interval was **−1.40 to +3.35 points**, and the observed clean-instruction loss exceeded the approximately 2-point tolerance. Therefore, I rejected the candidate rather than submit an unsupported improvement with a measured regression.

Reverting preserves baseline behavior; **it does not solve tag sensitivity**. Results and the decision are saved in `/work/final_decision.json` and `/work/notes.md`.