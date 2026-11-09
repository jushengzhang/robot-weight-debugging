**Final choice: keep `lex_dest_exact`, already restored in `WORK/edited`.**

### What changed
Only **20 rows** of `paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight` were edited: **40,822 coefficients** total.

The edits align selected descriptive tokens with existing task anchors, such as cook/cooker -> stove, grape/fermented -> wine, and unit/furniture -> cabinet. “Rounded” and “mixing” use half-strength blends toward bowl. All other tensors and all untargeted embedding rows remain exactly original. No training or inference changes were used.

### Mechanistic evidence
The failure was not simply insufficient language attention: failed prompts already received attention. Changing their **semantic embedding directions**, rather than their magnitudes, changed the selected actions:
- Direction-only controls reproduced the full repair’s action displacement with **0.983 cosine similarity**; norm-only controls did not.
- For “activate the cooktop,” action distance to the successful stove control fell from **2.409 to 0.417**.
- Removing the three destination substitutions reduced development success from **44/64 to 40/64**, losing four cabinet-placement successes involving “storage unit.”

### Validation
Across **1,533 paired trials on 577 non-development practice variants**, success increased from **1,356 to 1,376**:
- **88.45% -> 89.76%: +1.30 percentage points**
- **25 repairs, 5 regressions**

Infrastructure-error pairs were retried at the same seeds; retaining their initial failures still gives **+1.17 points**. Results are seed-dependent: the initial 1,152-trial validation gained **1.91 points** with variant-clustered SE **0.50**, while the additional 381-trial seed test lost **0.52 points**.

### Other perturbations
Across **1,518 unique non-language variant/seed pairs**, reconciled category differences were zero except Light Conditions at **-0.85 points**. The latest **768 difficult paired trials had no outcome discordances**. Fixed-input canonical probes were bit-identical.

I expect a modest improvement by connecting unfamiliar descriptions to already-grounded task features while preserving the unchanged canonical-instruction pathway. This is a targeted lexical repair, not a general solution to paraphrase robustness; the seed-dependent regressions remain a limitation.