# All 16 robot editing sessions

These are the robot editing sessions reported in the first research release of
[*Can LLMs Debug Robots by Editing Their Weights?*](https://arxiv.org/abs/2611.09376).
Each run has its own README, linking the editor's notes, final answer, executed
code, and tool feedback. Start with [language session C](pi_lang_c) for the
65-row edit, or [recovery session A](rec_a) for an unsuccessful search.

## Which run answers which question?

| Session | Policy / target | Final statement and records |
| --- | --- | --- |
| [Language repair A](pi_lang_a) | π₀.₅ | Retain the frozen 18-row edit. [Final answer](pi_lang_a/turns/t107/answer.md) |
| [Language repair B](pi_lang_b) | π₀.₅ | Keep the frozen lexical embedding edit. [Final answer](pi_lang_b/turns/t103/answer.md) |
| [Language repair C](pi_lang_c) | π₀.₅ | Submit the frozen 65-row embedding edit. [Final answer](pi_lang_c/turns/t108/answer.md) |
| [Instruction suffixes: first search](fw_suffix) | FastWAM | Withdrawal requested; final cleanup code was not executed. [Final answer](fw_suffix/turns/t112/answer.md) |
| [Instruction suffixes: follow-up A](fw_suffix2_a) | FastWAM | Original checkpoint submitted; cleanup verified in the final answer. [Final answer](fw_suffix2_a/turns/t147/answer.md) |
| [Instruction suffixes: follow-up B](fw_suffix2_b) | FastWAM | Original checkpoint submitted; all 308 candidate edits removed. [Final answer](fw_suffix2_b/turns/t128/answer.md) |
| [Instruction suffixes: follow-up C](fw_suffix2_c) | FastWAM | No validated repair claimed; rollback was not executed and the candidate remained on disk. [Final answer](fw_suffix2_c/turns/t142/answer.md) |
| [Camera perturbations](fw_camera) | FastWAM | Withdrawal requested; final cleanup code was not executed. [Final answer](fw_camera/turns/t127/answer.md) |
| [Background perturbations](fw_background) | FastWAM | Original checkpoint chosen; no value-changing edits retained. [Final answer](fw_background/turns/t147/answer.md) |
| [Language perturbations](fw_language) | FastWAM | Withdrawal requested; final cleanup code was not executed. [Final answer](fw_language/turns/t140/answer.md) |
| [Failure-state recovery A](rec_a) | π₀.₅ | Original checkpoint submitted; no retained edits. [Final answer](rec_a/turns/t165/answer.md) |
| [Failure-state recovery B](rec_b) | π₀.₅ | Original checkpoint submitted; no retained edits. [Final answer](rec_b/turns/t130/answer.md) |
| [Failure-state recovery C](rec_c) | π₀.₅ | Original checkpoint submitted; no retained edits. [Final answer](rec_c/turns/t148/answer.md) |
| [Broader robustness A](pro_a) | π₀.₅ | Rejection recommended; rollback still required in the final answer. [Final answer](pro_a/turns/t097/answer.md) |
| [Broader robustness B](pro_b) | π₀.₅ | Original checkpoint submitted; no active edits. [Final answer](pro_b/turns/t127/answer.md) |
| [Broader robustness C](pro_c) | π₀.₅ | Original checkpoint submitted; no active edits. [Final answer](pro_c/turns/t102/answer.md) |

All three language edits improve significantly on held-out data. The other
sessions explore the limits of repair; their final answers also expose the
difference between a rejected candidate and a rollback that actually ran.
The paper's cross-session accounting and held-out caveats are collected in
[results and interpretation](../docs/RESULTS.md).

## What the files contain

| File | Contents |
| --- | --- |
| `work/notes.md` | The editor's experiment notebook, including hypotheses and practice validation |
| `turns/tNNN/answer.md` | The assistant's full answer at a recorded turn |
| `tool_calls.jsonl` | Executed Python code, printed outputs, and tool requests/results |
| `work/*.json` | Saved diagnostics, candidate descriptions, and validation reports, where available |
| `state.json` | Model, effort, turn count, forward passes used, and budget |
| [run-manifest.json](run-manifest.json) | Original source locations and exported file counts |

The task specifications and rules are in [the experiment domains](../experiments/domains).
Held-out episode records and statistical audits are under [evidence](../evidence).
Original host paths are retained as provenance. Large activation dumps and model
arrays are omitted; their filenames remain in the logs. The evaluated language-C
embedding rows are separately included under [edits](../edits/pi05-language-c).

## Retention is a decision, not a filename

Read the final answer and notes before interpreting any saved candidate. In the
first FastWAM wave, final cleanup code was not executed even when withdrawal was
requested. Some other final messages also report an incomplete rollback. The
current driver includes the subsequent final-turn cleanup fix. The archived
answers and executed logs preserve what happened in each original run.

The existence of a file or a positive post hoc held-out result does not establish
that the editor retained the candidate or completed the repair.

[Repository home](../README.md) · [中文首页](../README_zh.md) · [Project status](../docs/PROJECT_STATUS.md)
