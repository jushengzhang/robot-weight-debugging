# Reproducing the release

## 1. Audit the reported results

From the repository root, run:

```bash
python3 evidence/audit_new_results.py
python3 -m pip install 'numpy>=1.26,<3'
python3 evidence/extended/audit_extended.py
```

These commands use released records and do not contact an LLM, start a simulator,
or consume GPU inference. The first script checks 660→704 successes in session C,
48 repairs/4 regressions, the 335/430 token partition, and the controls. The second
recomputes the extended evidence. The outputs are `evidence/new-results-verified.json`
and `evidence/extended/extended-evidence-verified.json`.

## 2. Obtain the evaluated base model

The main language result uses the released OpenPI `pi05_libero` checkpoint,
converted to PyTorch with OpenPI revision `215abfb` and flattened using the supplied
`experiments/backends/pi05/export_weights.py`. The evaluated parameter name is
`paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`,
shape 257152 × 2048. Preserve the normalization assets and the matching model config.

The [original backend environment](../experiments/backends/pi05/README.md)
records simulator versions, checkpoint conversion, observations, action chunking,
noise/seeding, and the eager inference setting. Compiled inference has different
numerics and was not used for the reported editing results. Exact compatibility
checks on the 65 original rows are required before applying the payload.

```bash
python3 edits/pi05-language-c/apply_sparse_rows.py \
  /path/to/pi05_libero_flat.safetensors --dry-run
python3 edits/pi05-language-c/apply_sparse_rows.py \
  /path/to/pi05_libero_flat.safetensors --output /path/to/embed_override.npy
```

The resulting `.npy` file is a replacement for the tensor named above. Register
it through the backend's `weights_override` mechanism; the script itself does not
launch inference. The base checkpoint is read-only and existing outputs are refused.

## 3. Configure the historical experiment layout

The release preserves the study sources under `experiments/`, including original
paths and launcher scripts for provenance. Do not run the historical shell
launchers unchanged: they include the authors' GPU allocation and network setup.

```bash
cp paths.example.json paths.local.json
# Edit workspace and the three Python paths for your installed environments.
python3 scripts/configure_workspace.py paths.local.json
source runtime/env.sh
```

The configuration command creates a new workspace and refuses to overwrite an
existing one. It copies Python sources with relocated paths, puts shared harness
modules in the locations expected by the domain imports, copies the practice
variant list, and records source/output hashes. It does not install dependencies
or make inference requests. `fastwam_work` is an optional path to the historical
FastWAM vendor layout described in its backend README; its default is
`<workspace>/vendor/fastwam-work`. Null Python paths use the configuring interpreter;
for actual rollouts, supply the appropriate installed environment executables.

The generated layout is:

```text
runtime/
  env.sh
  configured-files.json
  pi05/                  # π₀.₅ backend, venv/bin/python, harness/, bench/
  pi05_recovery/          # recovery backend; shares the π₀.₅ checkpoint
  pi05_pro/               # PRO backend; shares the π₀.₅ checkpoint
  study/harness/         # FastWAM domain and shared driver/kernel
  study/fastwam/         # FastWAM backend
  study/bench/           # practice variants
```

Install external dependencies and place checkpoint/assets as documented in the
backend READMEs. The configuration layer has been checked for source syntax and
domain imports; a fresh installation of every external simulator environment has
not been rerun as part of packaging this release.

## 4. Start one server explicitly

For example, after installing the π₀.₅ environment and dependencies:

```bash
source runtime/env.sh
mkdir -p runtime/pi05/run/sock
CUDA_VISIBLE_DEVICES=0 runtime/pi05/venv/bin/python \
  runtime/pi05/policy_server.py --sock runtime/pi05/run/sock/g0_0.sock \
  --ckpt /path/to/pi05_libero_pytorch
```

Choose a GPU available on your machine. Use the client/server protocol documented
in the backend README to check readiness and run paired episodes.

## 5. Run a new editing session

Once a compatible backend, flat weights, practice manifest, and kernel dependencies
are ready, provide your own `OPENAI_API_KEY` in the environment and invoke:

```bash
source runtime/env.sh
runtime/pi05/venv/bin/python runtime/pi05/harness/driver.py \
  --run runtime/pi05/runs_wsp/new_session \
  --domain runtime/pi05/harness/domain_pi05.py \
  --model gpt-6-astra --effort max --passes 50
```

This step makes paid LLM calls and simulator requests. The model name is the one
recorded in the study; availability depends on your provider/account. No API
credential is included in the release. Running a new session need not reproduce
the same edit; use the archived transcripts and the evaluated sparse payload for
the exact reported artifact.

Keep held-out evaluation separate from editing feedback. Retain the episode
lists, seeds, error accounting, and final retain/withdraw decision when comparing
results. Archived tensors alone do not establish submission intent.
