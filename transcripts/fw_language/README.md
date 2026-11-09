# Language perturbations

Can weight edits repair failures under changed instruction wording?

| Setting | Recorded value |
| --- | --- |
| Run | `fw_language` |
| Policy | FastWAM |
| Editor | `gpt-6-astra`, effort `max` |
| Forward-tool budget used | 40 / 40 |
| Final statement | Withdrawal requested; final cleanup code was not executed. |

## Read the session

1. [Experimental notes](work/notes.md): hypotheses, interventions, feedback, and validation.
2. [Final answer](turns/t140/answer.md): the editor's own concluding statement and submission intent.
3. [Every assistant answer](turns), starting with [the first reply](turns/t001/answer.md).
4. [Executed tool calls and outputs](tool_calls.jsonl).
5. [Saved diagnostic reports](work) and [run metadata](state.json).

The [task specification](../../experiments/domains/fastwam/domain_fastwam.py) describes the allowed
operations and feedback. The [shared driver](../../experiments/harness/driver.py)
and [Python kernel](../../experiments/harness/kernel.py) execute the loop.

## Interpreting the outcome

Read the final statement above together with the notes. Practice validation,
held-out evaluation, and the decision to retain or withdraw are separate records.
[Evaluation of the archived candidate](../../evidence/raw/fastwam/heldout/wave1/summary.md) is released for inspection.
A candidate left on disk does not by itself establish a completed repair.
See [results and limitations](../../docs/RESULTS.md) for the paper-level interpretation.

[All 16 sessions](../README.md) · [Reproduction guide](../../docs/REPRODUCING.md)
