# π0.5 language run C: sparse embedding edit

This payload contains 65 original and edited token-embedding rows, extracted from the evaluated archived override. It is not a full base checkpoint. The table is `paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`, shape 257152 × 2048. Exact elementwise comparison found 132768 changed scalar entries; no other tensors were submitted for this edit.

Use the same openpi `pi05_libero` base checkpoint converted to the flat safetensors layout used by the supplied backend. Install NumPy, then validate compatibility without creating a full tensor:

    python apply_sparse_rows.py /path/to/pi05_libero_flat.safetensors --dry-run

Create a new float32 NumPy override (about 2.11 GB):

    python apply_sparse_rows.py /path/to/pi05_libero_flat.safetensors --output /path/to/embed_override.npy

The script checks the shape and exact original values of all 65 rows, refuses to overwrite existing output files, and does not modify the original checkpoint. These row checks do not verify every other parameter in an arbitrary checkpoint. Use the checkpoint/configuration recorded in the experiment source. Register the output under the tensor name above using the provided backend's weight-override mechanism; this script does not modify inference code or launch a robot.

`embedding-recipe.json` lists all tokenizer IDs, destinations, interpolation strengths, and measured cosine values. The NPZ stores the actual evaluated edited values, avoiding reconstruction/rounding differences. Original rows originate from a BF16 export and are represented losslessly as float32 here.

Evaluation: 765 held-out language variants, seed 7; 660→704 successes, 48 repairs/4 regressions. Affected token subset: 249→293/335. Unaffected subset: 411→411/430. Context-insensitive edits can affect new uses of these tokens differently. Paper: https://arxiv.org/abs/2611.09376. Code: https://github.com/jushengzhang/robot-weight-debugging. Base-model terms continue to apply to these derived rows; see the repository third-party notices.
