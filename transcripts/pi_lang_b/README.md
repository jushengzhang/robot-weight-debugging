# Language repair B

Can editing token embeddings repair unfamiliar task wording?

| Setting | Recorded value |
| --- | --- |
| Run | `pi_lang_b` |
| Policy | π₀.₅ |
| Editor | `gpt-6-astra`, effort `max` |
| Forward-tool budget used | 50 / 50 |
| Final statement | Keep the frozen lexical embedding edit. |

## Read the session

1. [Experimental notes](work/notes.md): hypotheses, interventions, feedback, and validation.
2. [Final answer](turns/t103/answer.md): the editor's own concluding statement and submission intent.
3. [Every assistant answer](turns), starting with [the first reply](turns/t001/answer.md).
4. [Executed tool calls and outputs](tool_calls.jsonl).
5. [Saved diagnostic reports](work) and [run metadata](state.json).

The [task specification](../../experiments/domains/pi05/domain_pi05.py) describes the allowed
operations and feedback. The [shared driver](../../experiments/harness/driver.py)
and [Python kernel](../../experiments/harness/kernel.py) execute the loop.

## Held-out result

On the common 765-episode held-out language population (seed 7):
**660 → 673 successes**, **+1.7 percentage points**,
with 18 repairs and 5 regressions (exact paired p = 0.01062).
These results come from [held-out evaluation](../../evidence/raw/remote186/heldout/lang_all/summary.md),
not the practice validation figures quoted in the session's final answer.

[All 16 sessions](../README.md) · [Reproduction guide](../../docs/REPRODUCING.md)
