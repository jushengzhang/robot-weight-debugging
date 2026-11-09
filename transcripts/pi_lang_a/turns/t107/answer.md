## Final choice: retain the frozen 18-row edit

The submitted weights in `/work/edited` change only **18 rows** of `paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`.

I replaced selected synonym/descriptive-token embeddings with existing literal-task embeddings—for example, `basin -> bowl`, `cupboard -> cabinet`, `cooker -> stove`, `spread -> cheese`, and `cup -> mug`. All other rows and tensors remain unchanged. No training or global scaling was used. The complete recipe is in `WORK/candidate18_frozen.json`.

### Mechanism and causal evidence

For the diagnosed failures, the action expert already attended to the instruction, especially its newline summary. The problem was **task identity conveyed by unfamiliar wording**, rather than simply insufficient language attention.

The targeted replacements corrected that content:
- For a matched cream-cheese description, action RMS distance from the literal-instruction prediction fell from **0.276 to 0.021**.
- A controlled diagnostic dose test produced **0/8, 4/8, and 8/8 successes** at zero, half, and full substitution.
- Changing only the common embedding-mean direction did not reproduce the benefit; token-specific directions mattered.

### Language results

All results below are **practice validation**, not held-out evaluation. The candidate remained frozen during confirmation.

- **610 non-diagnostic variants, two seeds:** **1068/1220 -> 1104/1220**, or **87.5% -> 90.5%**. Gain: **2.95 percentage points**, with variant-clustered SE **0.71 points**; 42 rescues and 6 regressions.
- **Difficulty 5 within that evaluation:** **127/218 -> 159/218**, a **14.68-point gain**.
- **Separate third-seed check of all 204 difficulty-4/5 validation variants:** **144/204 -> 158/204**, gaining **6.86 points** (paired SE 2.03). Difficulty 4 alone was slightly negative; the benefit was concentrated at difficulty 5.

### Non-language safety

Across **1,024 paired episodes** with the final candidate:

| Category | Pairs | Success-rate change |
|---|---:|---:|
| Camera Viewpoints | 128 | 0.00 pp |
| Robot Initial States | 128 | 0.00 pp |
| Light Conditions | 256 | 0.00 pp |
| Background Textures | 192 | -0.52 pp |
| Sensor Noise | 192 | -0.52 pp |
| Objects Layout | 128 | 0.00 pp |

These retain all raw outcomes. Additionally, **40 canonical-input replays produced bit-identical action chunks and all 1,920 recorded activation arrays**.

This is a targeted lexical repair, not a general solution to arbitrary paraphrases. I expect it to help unseen descriptions using these cues while preserving canonical-instruction behavior, because the edited rows do not overlap the canonical instruction tokens audited.