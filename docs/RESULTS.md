# Results and interpretation — first release

This page summarizes the released simulation results. For version 2 and real-robot
work in progress, see [project status](PROJECT_STATUS.md).

## Language repair

Three independent editing sessions use the same 765 held-out LIBERO-Plus language
variants, with seed 7. The held-out population is separate from editing feedback.

| Session | Original successes | Edited successes | Change | Exact paired p |
| --- | ---: | ---: | ---: | ---: |
| [A](../transcripts/pi_lang_a) | 660 / 765 | 684 / 765 | +3.1 pp | 0.000182 |
| [B](../transcripts/pi_lang_b) | 660 / 765 | 673 / 765 | +1.7 pp | 0.0106 |
| [C](../transcripts/pi_lang_c) | 660 / 765 | 704 / 765 | +5.8 pp | 1.31 × 10⁻¹⁰ |

Session C changes 65 embedding rows, comprising 132,768 scalar changes
(0.00367% of policy parameters). Its 48 repairs and 4 regressions occur among
the 335 instructions containing edited token IDs: 249 → 293 successes,
or 74.3% → 87.5%. Among the other 430 episodes, both arms have 411 successes
and no paired success/failure changes. This last statement does not establish
identical trajectories. The [released payload](../edits/pi05-language-c) contains
the actual original and edited rows.

The editing process adds no gradient updates or new demonstration data. The
base policy already benefited from pretraining and robot demonstration fine-tuning;
the editor also receives budgeted simulator feedback.

## Controls

Matched random-direction edits do not help. Full replacement using the editor's
lexicon explains much of the gain but creates more regressions. Its direct
comparison with the editor is **not significant at 0.05** (exact paired p = 0.0614
on 763 common valid pairs). The released audit reports both requested-episode
and valid-pair error conventions, including failed evaluations.

Read the [control summaries](../evidence/raw/remote186/heldout/lang_ctrl/summary.md)
and the `controls` field of [the recomputed results](../evidence/new-results-verified.json).
The [extended evidence](../evidence/extended) covers seeds, locality, and
norm/direction probes.

## Unsuccessful searches and archived candidates

The study includes 16 robot editing sessions. All three language edits are
retained and improve significantly on held-out data. In the other 13 sessions,
the paper records 12 declined candidates and one remaining suffix follow-up
candidate without a significant held-out gain (p = 0.54). This is a description
of these searches, not a validated general failure classifier.

**Submission intent and files on disk must be read separately.** In the first
FastWAM wave, final cleanup code was not executed, leaving candidates archived
despite withdrawal requests. The driver includes the subsequent cleanup fix;
other archived final messages also explicitly report incomplete or unexecuted
rollback. A positive evaluation of an archived candidate does not turn it into
a completed, retained repair. Each [session README](../transcripts) links the
editor's final statement so this distinction can be inspected directly.

The suffix follow-up does not reproduce a significant repair. No recovery
improvement is claimed. LIBERO-PRO remains uncertain: the archived A candidate
has +1.5 pp overall on 1,484 perturbed episodes (p = 0.0448), while target-category
gains and the editor's own confirmation evidence do not establish a robust repair.
See [the PRO records](../evidence/raw/remote186/heldout_pro) and
[the archived decision](../transcripts/pro_a).

## A separate observation: metadata sensitivity

On 210 paired episodes, appending evaluation metadata reduces FastWAM success
from 70.0% to 48.6%. The evaluated π₀.₅ comparison has unchanged aggregate
success. This observation concerns input formatting; it does not imply that
a weight edit repaired the suffix problem. See the `suffix` field in
[the audit output](../evidence/new-results-verified.json).

## Recompute these results

From the repository root:

```bash
python3 evidence/audit_new_results.py
python3 -m pip install 'numpy>=1.26,<3'
python3 evidence/extended/audit_extended.py
```

The first command needs only Python's standard library. Both operate on released
records without inference. Full backend and editing instructions are in the
[reproduction guide](REPRODUCING.md). The first release contains no RoboDojo
result claim.
