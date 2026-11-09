# π0.5 (openpi `pi05_libero`, PyTorch) × LIBERO-Plus backend (host js186)

This is the second model for Weight-Space Programming. It mirrors the FastWAM backend on host 120
(`120:/mnt/SSD2_16TB/jusheng/wsp/fastwam`): same request protocol, client API, result schema and episode bookkeeping.
Everything lives in `/mnt/SSD_16TB/jusheng/wsp_pi05/` (`F` below). `~/wsp_pi05` is a symlink to it.

| path | what |
|---|---|
| `policy_server.py` | persistent server. 1 process = 1 π0.5 copy on 1 GPU, listening on unix socket `run/sock/g<gpu>_<k>.sock` |
| `client.py` | CLI + Python API (`import client as C`). Identical to the FastWAM client apart from the socket path, the `compile` flag and the reference column |
| `start_servers.sh` / `stop_servers.sh` | launch / stop detached servers. Only GPUs 0,1,6,7 are allowed. `ORIG=1` starts servers on original LIBERO; `EXTRA=` passes extra server args |
| `validate.py` | backend validation (results in `runs/validate/validation.json`) |
| `export_weights.py` → `weights/pi05_libero_flat.safetensors` + `weights/index.json` | flat export of the editable weights |
| `make_scan.py`, `analyze_scan.py` | weak-spot scan on the practice half (`bench/practice_variants.json` copied from `120:wsp/bench`) |
| `make_orig_sample.py` | unperturbed sanity sample on original LIBERO |
| `samples/` | `baseline210.jsonl`, `determinism5.jsonl`, `smoke.jsonl` (copied from the FastWAM backend); `scan700_practice.jsonl`; `orig_libero5.jsonl` |
| `ckpt/jax/pi05_libero` | official JAX checkpoint `gs://openpi-assets/checkpoints/pi05_libero` (12.4 GB). Every file checked against the GCS md5/crc32c (`ckpt/jax/listing_full.json`) |
| `ckpt/pi05_libero_pytorch` | converted with openpi `examples/convert_jax_model_to_pytorch.py --config_name pi05_libero --precision float32` (14.5 GB fp32). `assets/` (norm stats) is copied from the JAX checkpoint, because the converter looks one directory too high |
| `third_party/openpi` | Physical-Intelligence/openpi @ 215abfb (fresh clone, unmodified) |
| `third_party/LIBERO-plus` | the patched LIBERO-plus copy from 120 (sylvestf/LIBERO-plus@4976dc3 + OpenWAM compat patch). `libero/libero/assets` (9.7 GB, 448,799 files) is streamed from 120's extracted `assets.zip` over the campus link (netcat, 44 s) |
| `third_party/LIBERO-orig` | original LIBERO copied from `~/openpi/third_party/libero` on 186. Used only for the unperturbed sanity run |
| `third_party/LIBERO-plus-upstream` | fresh clone of upstream (also 4976dc3, so the copy matches) |
| `libero_cfg/`, `libero_cfg_orig/` | LIBERO configs, selected via `LIBERO_CONFIG_PATH`. `~/.libero` is untouched |
| `venv/` | Python 3.11 venv, see "Environment" |
| `stubs/lerobot` | stub of `lerobot.common.datasets.lerobot_dataset`. openpi imports it only for training data loading |
| `cache/` | uv/pip/HF/openpi caches (PaliGemma tokenizer in `cache/openpi/big_vision`) |

## Environment
* `venv`: Python 3.11.13 (uv).
  * **torch 2.12.0+cu130**, copied from conda env `nbcg` (torch, triton, nvidia-*, cuda-*). The 186 internet link was ~1-2 MB/s at the time. 2.12 + cu130 supports sm_120.
  * transformers 4.53.2 + openpi `transformers_replace` patch (checked by openpi's own check).
  * jax/jaxlib 0.5.3 **CPU only** (used for the conversion and openpi's transforms; JAX_PLATFORMS=cpu).
  * flax 0.10.2, orbax 0.11.13.
  * mujoco 3.3.2, robosuite 1.4.1, bddl 1.0.1, gym 0.25.2, numpy 1.26.4, Wand 0.6.13, scikit-image 0.25.2. These are the same sim versions as the FastWAM env on 120.
  * openpi, openpi-client and LIBERO-plus are installed `-e --no-deps`.
  * Install scripts: `install_env.sh` (first attempt, abandoned) and `install_env2.sh` (used).
* ImageMagick (Sensor-Noise variants): `MAGICK_HOME=~/anaconda3/envs/erase` (IM 7.1.2-0, read-only use). On 120 it is IM 7.1.2-31.
* Rendering: `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=<gpu>`.

## Starting / stopping servers
```bash
cd /mnt/SSD_16TB/jusheng/wsp_pi05
bash start_servers.sh "0:6 1:6 6:6 7:6"      # 24 servers g<gpu>_<k>; ready when run/sock/<name>.ready exists (~40-60 s)
START=6 bash start_servers.sh "0:2"          # add g0_6, g0_7 (START = first index)
ORIG=1 bash start_servers.sh "6:2"           # o6_0, o6_1 on ORIGINAL LIBERO (sanity)
venv/bin/python client.py ping --socks 'run/sock/*.sock'
bash stop_servers.sh                         # kills only processes matching wsp_pi05/policy_server.py
```
**Memory and packing.**
* One process uses 7.5 GB after load and 7.8 GB max reserved during rollouts.
* Tested at 6-8 processes per RTX 6000D (84 GB), plus 26 servers at once on 4 GPUs. The simulator, not the GPU, is the bottleneck: inference is ~0.1 s per query.
* With the 320-core host at load ~15-20, 24 servers did 210 episodes in 7.2 min (~29 episodes/min).
* Load time ~40 s.

## Protocol (same requests as FastWAM)
`{"cmd": "run"|"probe"|"ping"|"info"|"params"|"shutdown", ...}`, with one JSON line per request. The `run` and `probe` fields are as in
`wsp/fastwam/README.md`: `episodes`, `weights_override`, `override_mode` (replace|add|scale), `arms` (orig/edit),
`record`, `save_frames`, `save_video`, `language` (strip|raw|clean), `compile`, `out_dir`. The Python API is unchanged:
`C.rollout(...)`, `C.probe(...)`, `C.call(...)`, `C.summarize(...)`. Episode results have the same fields
(`success, steps, termination, failure, n_queries, instruction, instruction_raw, instruction_clean, action_sha1,
actions, wall_s, restore_verified, ...`).

Differences from FastWAM:
* **`compile` defaults to false (eager).**
  * `compile=true` wraps `sample_actions` in `torch.compile` (mode `default`). It works: a 220-step episode takes 11 s instead of ~14 s, and the first call costs ~50 s. It is deterministic across repeats.
  * Its numerics differ from eager. The test episode failed compiled but succeeded eager.
  * Weight edits under compile are not validated. Use eager, which is the default and the mode of every result below.
* Actions are executed exactly as openpi outputs them: the first 7 dims, after openpi's Unnormalize. There is no gripper binarization, which is upstream behaviour.
* `probe` result: `action` (.npy [10,7], executed space), `matches_saved_chunk`, `activations`, `attention`, `prefix_layout`. There is no `action_norm`.

### Episode settings (upstream openpi `examples/libero/main.py`)
* Env: `OffScreenRenderEnv(bddl, 256x256)`, `env.seed(seed)`, `init_states[init_idx]`. Seed 7, init_idx 0 (LIBERO-Plus protocol: 1 trial per variant).
* Timing: 10 dummy settle steps (`[0]*6+[-1]`). Max steps: 220 spatial, 280 object, 300 goal, 520 libero_10. These are openpi's values, not FastWAM's 400/700.
* Observation:
  * agentview and wrist images are rotated 180°, then `resize_with_pad` to 224.
  * state = eef_pos + axis-angle(eef_quat) + gripper_qpos (8-d).
  * `prompt` = the instruction.
* Replanning: **action chunk 10 (`pi05_libero` action_horizon), execute the first 5 (replan_steps=5)**, 10 flow-matching steps (openpi default).
* Noise: `torch.Generator(cpu).manual_seed(seed)` per episode, with a fresh `randn(10, 32)` per query. Upstream draws from the global RNG, so this is deterministic but otherwise equivalent.
* `random`, `np.random` and `torch` are all seeded with `seed` at episode start.
* `language`:
  * `strip` (default) = the original LIBERO instruction.
  * `raw` = `task.language` with the LIBERO-Plus suffix, which is what the reference `main_plus.py` runners feed.
  * `clean` = the BDDL language.

## Weight overrides
* Mechanics are the same as FastWAM: a dir/file of `<name>.npy` / `.safetensors` / `.npz`, cast to the parameter dtype and written in place.
* The pristine copy is kept on CPU. Restore is verified with `torch.equal` plus sha256, giving `restore_verified`, and the request fails if verification fails.
* Accepted names:
  * full parameter names (below);
  * the aliases `vlm.` = `paligemma_with_expert.paligemma.model.language_model.`, `expert.` = `paligemma_with_expert.gemma_expert.model.`, `vision.` = `paligemma_with_expert.paligemma.model.vision_tower.vision_model.`, `projector.` = `paligemma_with_expert.paligemma.model.multi_modal_projector.`;
  * an optional leading `model.`.
* `python client.py params --prefix expert.layers.0. --socks run/sock/g0_0.sock` lists names, shapes and dtypes.

## Parameter naming (3.617 B params, 812 tensors; `weights/index.json`)
| group | prefix | params | layout |
|---|---|---|---|
| vision | `paligemma_with_expert.paligemma.model.vision_tower.vision_model.` | 412.4 M | SigLIP So400m/14: `embeddings.{patch_embedding,position_embedding}`, `encoder.layers.{0..26}.{self_attn.{q,k,v,out}_proj, layer_norm1/2, mlp.fc1/fc2}`, `post_layernorm`; width 1152, 256 tokens/image |
| projector | `paligemma_with_expert.paligemma.model.multi_modal_projector.linear` | 2.4 M | 1152→2048 |
| vlm | `paligemma_with_expert.paligemma.model.language_model.` | 2508.5 M | Gemma-2B: `embed_tokens` (257152×2048, tied with `paligemma.lm_head`, which is unused), `layers.{0..17}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}`, `norm`; 8 heads, 1 KV head, head_dim 256, mlp 16384 |
| expert | `paligemma_with_expert.gemma_expert.model.` | 427.9 M | Gemma-300M action expert: `layers.{0..17}.{self_attn.{q,k,v,o}_proj (8 heads x 256, 1 KV head; q 1024->2048), mlp.{gate,up,down}_proj, input_layernorm.dense, post_attention_layernorm.dense}` (adaRMS: the norms are Linear(1024→3072) conditioned on the flow time), `norm.dense`; width 1024, mlp 4096, shares attention with the VLM KV cache |
| expert_lm_head (unused) | `paligemma_with_expert.gemma_expert.lm_head.weight` | 263.3 M | never used at inference |
| action heads | `action_in_proj` (32→1024), `action_out_proj` (1024→32), `time_mlp_in`, `time_mlp_out` | 2.2 M | fp32 |

* dtypes at inference (= the export): 690 bf16 and 122 fp32 tensors. The fp32 ones are the layernorms, the SigLIP patch/position embeddings and the action heads, following openpi `to_bfloat16_for_selected_params`.
* `weights/pi05_libero_flat.safetensors` holds exactly these tensors, deduplicated. The tied `paligemma.lm_head.weight` is not stored.
* Inference per query:
  1. SigLIP encodes 3 images: base, left wrist, and a right-wrist image of zeros that is masked out.
  2. The prefix is 3×256 image tokens + 200 prompt tokens (padded). π0.5 with `discrete_state_input=False` does **not** use the proprio state.
  3. The Gemma-2B prefill builds the KV cache.
  4. The action expert runs 10 Euler denoising steps over 10 action tokens.

## Activation recording / attention (eager, the same numerics as rollouts)
* `record` spec: `{"vlm_layers": [...], "expert_layers": [...], "vision_layers": [...], "what": ["attn_out","mlp_out","resid"], "per_call": false}`, and/or `{"modules": [raw or aliased module names]}`.
* Meaning of `what`: `attn_out` = `layers.i.self_attn` output (after o_proj); `mlp_out` = `layers.i.mlp` output; `resid` = decoder-layer output.
* Keys and shapes:
  * vlm: `mean::vlm.<i>.<what>::{base,wrist,lang}`, [2048] (means over the 256 image tokens of each camera and over the valid prompt tokens), 1 call per query.
  * expert: `mean::expert.<i>.<what>::action`, [1024], 10 calls per query.
  * vision: `mean::vision.<i>.<what>::{base,wrist}`, [1152], 1 call per image per query.
  * Plus `ncalls::...` counts, and `calls::...` when `per_call` is true.
* Probe attention (`attention_layers`: expert layers, at most 4 inputs × 4 layers): softmax weights of the 10 action queries over the 968 prefix keys, averaged over the 10 denoising steps, giving `[8 heads, 10, 968]`. `prefix_layout` gives the token ranges.
* Example (layer 9, libero_spatial step 0): 77% of the mass goes to the prefix and the rest to action tokens. The split is base 7.7%, wrist 44.7%, lang 24.7%, masked right-wrist 0.

## save_frames
Inputs are saved at the first query and at steps 0, 50, 100, ... when the step aligns with a query.
* `<stem>_frames/tNNNN.png` (agent | wrist, 224×448).
* `tNNNN_inputs.pt` = `{element (the exact openpi observation dict: 224² uint8 images, 8-d state, prompt), noise [10,32], seed, suite, variant_id, step, arm, action_chunk [10,7], replan_steps}`.
* `C.probe([...])` replays them through `policy.infer(element, noise=noise)`.

## Validation (`venv/bin/python validate.py run/sock/g0_9.sock run/sock/g1_9.sock`; `runs/validate/validation.json`)
* **Determinism:** same 5 episodes (`determinism5`) twice in one process gave 5/5 bit-identical action traces (`action_sha1`); across processes (GPU0 vs GPU1) also 5/5. In stage 1, the 30 Language variants (same instruction under strip and raw) were identical across the two runs and servers: 30/30.
* **Identity override** (`action_out_proj.weight` and `vlm.layers.5.self_attn.o_proj.weight` set to their own values; paired orig/edit): edit == orig 5/5, `restore_verified` true.
* **Destructive overrides:**
  * `action_out_proj.weight = 0` changed 5/5 traces; success 0/5 (orig 4/5).
  * Gaussian 0.5σ noise on `expert.layers.10.mlp.down_proj.weight` changed 5/5 traces; success 4/5 (robust).
  * An orig run after the edits was bit-identical to the pre-edit run, 5/5.
* **Record:** a run with hooks has actions identical to a plain run (2/2). 84 arrays. `ncalls` = 20 per query-level hook and 200 for the expert (20 queries × 10 steps).
* **save_frames + probe:** probe(orig) on the saved inputs reproduces the executed chunk exactly (2/2). The edit arm gives different actions (max |Δa| 0.0027).

## STAGE 1 results (original weights, eager, seed 7, init_idx 0)
`samples/baseline210.jsonl` (the FastWAM baseline sample: 30 per category), 24 servers.
`runs/baseline210_strip/`, `runs/baseline210_raw/`.

| category | n | **strip** | raw | published π0 (LIBERO-Plus Table 2) | FastWAM strip / raw (120) |
|---|---|---|---|---|---|
| Camera Viewpoints | 30 | **83.3** | 80.0 | 13.8 | 46.7 / 16.7 |
| Robot Initial States | 30 | **83.3** | 80.0 | 6.0 | 76.7 / 33.3 |
| Language Instructions | 30 | **83.3** | 83.3 | 58.8 | 63.3 / 63.3 |
| Light Conditions | 30 | **96.7** | 100.0 | 85.0 | 90.0 / 76.7 |
| Background Textures | 30 | **90.0** | 93.3 | 81.4 | 63.3 / 50.0 |
| Sensor Noise | 30 | **96.7** | 96.7 | 79.0 | 60.0 / 33.3 |
| Objects Layout | 30 | **86.7** | 86.7 | 68.9 | 90.0 / 66.7 |
| **TOTAL** | 210 | **88.6** | 88.6 | 53.6 | 70.0 / 48.6 |

| suite | n | strip | raw |
|---|---|---|---|
| libero_spatial | 52 | 92.3 | 94.2 |
| libero_object | 53 | 88.7 | 90.6 |
| libero_goal | 53 | 84.9 | 86.8 |
| libero_10 | 52 | 88.5 | 82.7 |

* **The variant suffix does not hurt π0.5** (unlike FastWAM). Of the 180 variants whose instruction differs, 9 flip S→F and 9 flip F→S.
* **No published π0.5 numbers:** the LIBERO-Plus README leaderboard (repo @4976dc3) and the paper (arXiv 2510.13626, Tables 1/2) contain π0 and π0-FAST but **no π0.5 row**. The π0 row is shown for reference.
* **Wall time per episode** (24 servers at load ~15-20): median 18.0 s, mean 23.9 s, max 279 s (Sensor Noise, CPU-bound corruption). Per-category medians (s): Background 23.0, Camera 15.0, Language 16.3, Light 15.2, Layout 17.2, Robot 14.6, Noise 36.4. Mean 36 queries per episode.

**Sanity on original LIBERO** (`runs/orig_libero5`: 10 tasks × init states 0-4 per suite, seed 7, 200 episodes):

| suite | ours | published openpi pi05_libero |
|---|---|---|
| spatial | 96 | 98.8 |
| object | 100 | 98.2 |
| goal | 98 | 98.0 |
| libero_10 | 88 | 92.4 |
| avg | **95.5** | 96.85 |

The differences are within n=50 noise.

## Weak-spot scan (practice half; `runs/scan700_practice_strip/scan_analysis.md`)
* Sample: 100 practice variants per category, stratified 20 per difficulty 1-5 (`random.Random(0)`). 59 Light variants with no difficulty label are excluded.
* Settings: seed 7, init_idx 0, strip, `--save-video`. Result: **86% (604/700)**, 0 errors, 16.7 min on 24 servers.

| category | d1 | d2 | d3 | d4 | d5 | all |
|---|---|---|---|---|---|---|
| Background | 100 | 95 | 100 | 90 | 85 | 94 |
| Camera | 100 | 95 | 85 | 70 | **40** | 78 |
| Language | 95 | 90 | 95 | 75 | **55** | 82 |
| Light | 95 | 100 | 95 | 95 | 85 | 94 |
| Layout | 85 | 90 | 95 | 85 | 80 | 87 |
| Robot | 90 | 90 | 85 | 70 | 65 | 80 |
| Noise | 100 | 100 | 95 | 85 | 65 | 89 |

By suite: spatial 93, object 92, goal 82, libero_10 79. The weakest category × suite cells are **Camera × libero_10 53% (16/30)** and **Language × libero_goal 56% (14/25)**.

Failure modes:
* Language/goal failures are **wrong task or wrong object**: paraphrases ("activate the cooktop", "fermented grape beverage", "dish above the cupboard") get mapped onto another goal task in the same scene.
* Camera d4-5 failures are geometric: partial completion of 2-object libero_10 tasks, or holding the object near the target.
* All failures are max-step timeouts. The dominant gripper pattern is a re-grasp loop.

## Things that failed / notes
* The 186 internet link was very slow and bursty (~1-2 MB/s for long stretches; the proxy sometimes stalled to KB/s, then recovered to 10+ MB/s). Long ssh sessions to 186 were dropped several times, so everything is launched with `setsid nohup`.
* LIBERO-Plus assets: the direct 120→186 link (campus netcat, ~200 MB/s) worked best. Going through the local machine ran at ~1 MB/s.
* The torch download timed out on pypi.nvidia.com, so torch was copied from a local conda env.
* The lerobot pi05 checkpoints on 120 (qinhan `pi05_libero_finetuned_v044`) are a lerobot re-finetune (chunk 50) and were not used.
* `install_env.sh` is the abandoned first attempt; `install_env2.sh` is the one actually used.
