# Broader robustness A

Can weight edits improve robustness on LIBERO-PRO perturbations?

| Setting | Recorded value |
| --- | --- |
| Run | `pro_a` |
| Policy | π₀.₅ |
| Editor | `gpt-6-astra`, effort `max` |
| Forward-tool budget used | 40 / 40 |
| Final statement | Rejection recommended; rollback still required in the final answer. |

## Read the session

1. [Experimental notes](work/notes.md): hypotheses, interventions, feedback, and validation.
2. [Final answer](turns/t097/answer.md): the editor's own concluding statement and submission intent.
3. [Every assistant answer](turns), starting with [the first reply](turns/t001/answer.md).
4. [Executed tool calls and outputs](tool_calls.jsonl).
5. [Saved diagnostic reports](work) and [run metadata](state.json).

The [task specification](../../experiments/domains/pi05/domain_pro.py) describes the allowed
operations and feedback. The [shared driver](../../experiments/harness/driver.py)
and [Python kernel](../../experiments/harness/kernel.py) execute the loop.

## Interpreting the outcome

Read the final statement above together with the notes. Practice validation,
held-out evaluation, and the decision to retain or withdraw are separate records.
[Evaluation of the archived candidate](../../evidence/raw/remote186/heldout_pro/pro_a_big/summary.md) is released for inspection.
A candidate left on disk does not by itself establish a completed repair.
See [results and limitations](../../docs/RESULTS.md) for the paper-level interpretation.

[All 16 sessions](../README.md) · [Reproduction guide](../../docs/REPRODUCING.md)
