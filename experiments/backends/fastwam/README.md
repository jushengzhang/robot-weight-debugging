# FastWAM × LIBERO-Plus forward-pass backend (host 120)

Backend for Weight-Space Programming on FastWAM: a **forward pass** = a batch of simulated LIBERO-Plus episodes
(original and/or edited weights, paired on identical variant + seed), optionally with activation capture, saved
policy inputs, and a simulation-free **probe** (actions / activations / attention maps).

Everything lives in `/mnt/SSD2_16TB/jusheng/wsp/fastwam/` (`F` below). Nothing under the Codex `work/` tree was modified.

| path | what |
|---|---|
| `policy_server.py` | persistent server: 1 process = 1 FastWAM copy on 1 GPU, unix socket `run/sock/g<gpu>_<k>.sock` |
| `client.py` | CLI + Python API: sharding over servers, aggregation (`results.jsonl`, `summary.md`), probe |
| `start_servers.sh` / `stop_servers.sh` | launch / stop detached servers |
| `make_sample.py` | stratified LIBERO-Plus sample (N per category, spread over the 4 suites) |
| `validate.py` | backend validation (override exactness, determinism, record, frames, probe) |
| `ckpt/libero_uncond_2cam224.pt` (+ `_dataset_stats.json`, `SHA256SUMS`) | `yuanty/fastwam` uncond checkpoint, sha256 verified against HF LFS oid |
| `third_party/LIBERO-plus` | copy of sylvestf/LIBERO-plus@4976dc3 + OpenWAM `libero-plus-compatibility.patch` (torch.load weights_only, thread-safe motion-blur); `assets` → symlink to `work/new_benchmark_assets/libero_plus/extracted/assets` |
| `libero_cfg/config.yaml` | LIBERO config (used via `LIBERO_CONFIG_PATH`; `~/.libero` is untouched) |
| `/mnt/SSD2_16TB/jusheng/wsp/envs/fastwam_plus` | new venv (openvla base python) layering `work/fastwam_runtime` site-packages (torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.1, fastwam editable) + `work/new_benchmark_runtime/libero_plus` (wand, scikit-image) + the LIBERO-plus copy, via `wsp_layers.pth` |
| `samples/baseline210.jsonl`, `runs/baseline210*/` | baseline sample and results |

## Starting / stopping servers

```bash
cd /mnt/SSD2_16TB/jusheng/wsp/fastwam
bash start_servers.sh "2:5 3:5"          # 5 servers on GPU 2, 5 on GPU 3 (names g2_0..g2_4, g3_0..g3_4)
START=5 bash start_servers.sh "0:2"      # add g0_5, g0_6 on GPU 0 (START = first index, avoid name clashes!)
ls run/sock/*.ready                      # a server is ready when its .ready file exists (~2-3 min load)
python client.py ping                    # all sockets
python client.py info --socks run/sock/g2_0.sock
bash stop_servers.sh                     # kills only processes matching wsp/fastwam/policy_server.py
```
Each server sets `CUDA_VISIBLE_DEVICES=<gpu>`, `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=<gpu>`, `LIBERO_CONFIG_PATH`, `MAGICK_HOME`
(ImageMagick for Sensor-Noise variants). Logs: `logs/server_g<gpu>_<k>.log`.

**Memory / packing:** one process = 25.4 GB reserved (model 24.9 GB: video DiT 5.0B + action DiT 1.02B + umT5-xxl 5.68B
+ VAE 0.70B, bf16) and stays flat during rollouts (CUDA graphs captured once). **5 processes per H200 (141 GB)**, 3 per
RTX Pro 6000 (96 GB). Load time ≈ 110 s (≈ 2-3 min when 10 load at once). The first compiled episode per process
includes ~20-40 s of torch.compile (cache in `cache/inductor`).

## CLI

```bash
python make_sample.py --per-cat 30 --seed 7 --out samples/my.jsonl
python client.py run --episodes samples/my.jsonl --out runs/my_run [--socks 'run/sock/g2_*.sock,run/sock/g3_0.sock']
       [--override DIR --arms orig,edit --override-mode replace|add|scale] [--language strip|raw|clean]
       [--save-frames 50] [--save-video] [--record rec.json] [--no-compile] [--per-request 1]
python client.py summarize --out runs/my_run
python client.py params --prefix action_expert.blocks.0.     # parameter names / shapes / dtypes
```
Results are appended to `<out>/results.jsonl` as episodes finish (re-running the same command resumes and skips
finished keys); per-category / per-suite table in `<out>/summary.md`; artifacts in `<out>/artifacts/`.

## Python API (`import sys; sys.path.insert(0, F); import client as C`)

```python
C.list_socks(pattern="/mnt/SSD2_16TB/jusheng/wsp/fastwam/run/sock/*.sock") -> list[str]

C.rollout(episodes: list[dict], out: str|Path, socks: list[str]|None = None,
          weights_override: str|Path|None = None, arms: list[str]|None = None,      # subset of ["orig","edit"]
          override_mode: str = "replace",                                           # "replace" | "add" | "scale"
          record: dict|None = None, save_frames: int = 0, save_video: bool = False,
          language: str = "strip", per_request: int = 1, log=print) -> list[dict]
    # episodes: [{"suite": "libero_goal", "variant_id": 1162, "seed": 7, "init_idx": 0(optional), "max_steps": (optional)}]
    # variant_id = id in LIBERO-Plus task_classification.json (1-based; task index = id-1, task_order_index 0)
    # default arms = ["edit"] if weights_override else ["orig"]; with ["orig","edit"] each episode runs under both,
    # same (variant, seed). Returns one dict per (episode, arm) -- see "Episode result" below.

C.probe(inputs: list[str|Path], out: str|Path, sock: str|None = None,
        weights_override=None, arms=None, override_mode="replace",
        record: dict|None = None, attention_layers: list[int]|None = None, compile: bool = False) -> dict
    # inputs = *_inputs.pt files written by save_frames; no simulation. Returns {"ok", "results": [...], "override": {...}}
    # results[i] = {"arm","input","idx","action_norm": .npy [32,7] (model space), "action": .npy [32,7] (executed space),
    #               "activations": .npz (if record), "attention": {layer: .npy [24 heads, 32 action tokens, 98 video tokens]},
    #               "video_grid_fhw": [1,7,14]}

C.call(sock, request_dict) -> dict            # raw protocol (one JSON line in, one JSON line out)
C.summarize(out) -> str                       # markdown table
```

### Raw request format (unix socket, one JSON line)
```json
{"cmd": "run",
 "episodes": [{"suite": "libero_spatial", "variant_id": 2026, "seed": 7}],
 "weights_override": "/abs/dir",  "override_mode": "replace", "arms": ["orig", "edit"],
 "record": {"action_blocks": [0, 15, 29], "video_blocks": [29], "what": ["attn_out", "mlp_out", "resid"], "per_call": false},
 "save_frames": 50, "save_video": false, "language": "strip", "compile": true,
 "out_dir": "/abs/out"}
```
Other commands: `{"cmd":"ping"}`, `{"cmd":"info"}`, `{"cmd":"params","prefix":"..."}`,
`{"cmd":"probe","inputs":[...],"out_dir":...,"attention":{"layers":[...]}, ...}`, `{"cmd":"shutdown"}`.

### Weight overrides
* `weights_override` = a directory (or single file) with `<param name>.npy` (any float dtype, cast to the param dtype
  bf16), `*.safetensors` and/or `*.npz` (keys = param names). Names accepted in any of these forms (all alias the same tensor):
  * checkpoint keys as saved in `libero_uncond_2cam224.pt["mot"]`: `mixtures.video.blocks.3.ffn.0.weight`, `mixtures.action.head.weight`
  * model names: `video_expert.blocks.3.ffn.0.weight`, `action_expert.head.weight`, `mot.mixtures...`, `dit.mixtures...`
  * `proprio_encoder.weight` / `.bias` (stored under `proprio_encoder` in the checkpoint), and also `text_encoder.*`, `vae.*`.
* `override_mode`: `replace` (default), `add` (orig + delta), `scale` (orig * factor).
* Mechanics: the pristine tensor is copied to CPU on first touch; the new value is written **in place** (`param.data.copy_`,
  same storage, so the compiled CUDA graphs see it); after each (episode, edit-arm) the original is copied back and
  verified exactly (`torch.equal` vs the CPU copy + sha256 of the bytes) → `restore_verified` in each result; the request
  fails loudly if verification fails. Each request is self-contained: the server is always back at original weights
  between requests.

### Episode protocol (matches upstream `experiments/libero/eval_libero_single.py` and LIBERO-Plus reference runners)
* `env.seed(seed)`, `random/np.random/torch` seeded with `seed` at episode start (Sensor-Noise corruptions use `np.random`),
  `init_states[init_idx]` (LIBERO-Plus protocol: 1 trial per variant, `init_idx=0`, `seed=7`).
* 30 no-op settle steps, replan every 10 steps with a 32-step chunk, gripper binarized; max steps 400
  (spatial/object/goal) / 700 (libero_10) — the FastWAM upstream values.
* flow-matching noise: `torch.Generator(cpu).manual_seed(seed)` for **every** policy query (upstream uses the constant
  `cfg.seed` the same way); 10 denoising steps, `compile_action_infer=True` (CUDA graphs) unless `compile=false`.
* `language`: **`strip` (default)** = `task.language` with the variant suffix removed (= the original LIBERO
  instruction; reproduces the published FastWAM numbers, 70.0% vs 70.8%). `raw` = `task.language` exactly as the LIBERO-Plus reference runners (openpi `main_plus.py`,
  OpenWAM `single_eval.py`) feed it — for non-language variants this **contains the variant suffix**
  (e.g. "...on the plate view 69 0 100 0 0 initstate 0", "... light 39", "... noise 10");
  this drops FastWAM to 48.6% (see baseline);
  `clean` = `env.language_instruction` from the BDDL (warning: for some variants this is a paraphrase).

### Episode result (one JSON per episode × arm)
`suite, variant_id, seed, init_idx, arm, name, category, difficulty, success, steps (policy steps), termination
("success" | "max_steps" | "error"), failure (None | "timeout" | "error: ..."), traceback (on error), n_queries,
instruction (fed to the model), instruction_raw, instruction_clean, action_sha1 (hash of the executed action trace),
actions (.npy [steps,7]), wall_s, restore_verified (edit arm), server`, and optionally
`activations` (.npz), `activation_calls`, `frames_dir`, `frames_steps`, `video` (.mp4, agent|wrist, every 2nd step).

### Activation recording (`record`, forces eager mode)
* Spec: `{"action_blocks": [...], "video_blocks": [...], "what": [...]}` with `what` ∈
  `attn_out` (= `blocks.i.self_attn.o` output, the projected mixed-attention output), `mlp_out` (= `blocks.i.ffn` output),
  `resid` (= block output / residual stream after the block = 2nd call of `blocks.i.gate`). Raw module names are also
  accepted via `"modules": [...]` (MoT never calls `block.forward`, so hook submodules such as `self_attn.q`, `ffn.0`, `cross_attn`).
* Summary = mean over all hooked calls in the episode (action expert: every policy query × 10 denoising steps;
  video expert: one first-frame prefill per query) of the output averaged over tokens:
  * action expert → `mean::action.<i>.<what>::action` (avg over the 32 action tokens), shape [1024]
  * video expert → `mean::video.<i>.<what>::agent` and `::wrist` (avg over the first-frame latent tokens of each camera;
    latent grid 1×7×14, agent = left 7 columns, wrist = right 7), shape [3072]
  * `ncalls::...` counts; with `per_call: true` also `calls::...` [n_calls, D].
  Saved as `<out>/artifacts/<suite>_v<id>_s<seed>_i<init>_<arm>_acts.npz` (np.load → dict of .npy arrays).
* Recording runs eager (hooks do not fire inside CUDA-graph replays). Eager and compiled actions are numerically
  slightly different (see validation), so compare orig vs edit **within the same mode**.

### save_frames (replayable inputs)
`save_frames: 50` → for the first policy query and every query at policy step 0, 50, 100, ... :
`<stem>_frames/tNNNN.png` (agent | wrist, 256×512) and `<stem>_frames/tNNNN_inputs.pt` =
`{prompt, input_image [1,3,224,448] bf16 in [-1,1], proprio [1,8] (normalized), seed, action_horizon, num_inference_steps,
suite, variant_id, step, arm, action_chunk (executed [32,7])}` — exactly the arguments passed to `model.infer_action`;
`C.probe([...])` replays them.

### Attention maps (probe only, ≤4 inputs × ≤4 layers, eager)
Mixed-attention probabilities of the action-expert queries over the first-frame video keys,
softmax(q·kᵀ/√d + mask) computed alongside the real attention, averaged over the 10 denoising steps, per head:
`[24 heads, 32 action tokens, 98 video tokens]` (video token t = row t//14, col t%14 of the 7×14 grid; cols 0-6 agent,
7-13 wrist). The remaining probability mass (1 − row sum) goes to action tokens.

## Parameter naming (FastWAM uncond, 12.41 B params, bf16)

| top-level module | params | notes |
|---|---|---|
| `video_expert` (= `mot.mixtures.video`, `dit.mixtures.video`) | 4,999,787,712 | Wan2.2-TI2V-5B DiT: 30 blocks, hidden 3072, 24 heads×128, ffn 14336; `patch_embedding`, `text_embedding.{0,2}`, `time_embedding.{0,2}`, `time_projection.1`, `blocks.{0..29}.{self_attn.{q,k,v,o,norm_q,norm_k}, cross_attn.{q,k,v,o,norm_q,norm_k}, norm3, ffn.{0,2}, modulation}`, `head` |
| `action_expert` (= `mot.mixtures.action`) | 1,020,886,023 | ActionDiT: 30 blocks, hidden 1024, 24 heads×128 (shared attention with video tokens), ffn 4096; `action_encoder`, `text_embedding.{0,2}`, `time_embedding.{0,2}`, `time_projection.1`, `blocks.{i}.…` (same layout), `head` (Linear 1024→7) |
| `text_encoder` | 5,680,910,336 | umT5-xxl encoder (frozen in training; prompt → 4096-d context) |
| `vae` | 704,688,668 | Wan2.2 VAE (only the encoder is used at inference: first frame → 48-ch latent) |
| `proprio_encoder` | 36,864 | Linear 8→4096, appended as one context token |

Checkpoint `libero_uncond_2cam224.pt` = `{"mot": 1649 tensors (video+action experts, keys mixtures.{video,action}.*), "proprio_encoder", "step", "torch_dtype"}`;
the 1649 keys map 1:1 onto all expert parameters (verified). text encoder / VAE come from the shared Wan files.
Inference per policy query: VAE-encode the current 2-camera frame (224×448) → video expert prefills a KV cache over
98 first-frame tokens (1 pass) → action expert denoises 32×7 actions in 10 steps attending to [video cache, action tokens].
`python client.py params --prefix action_expert.blocks.0.` lists exact names/shapes.

## Baseline (FastWAM uncond, original weights)

Sample: `samples/baseline210.jsonl` = 30 variants per category (7-8 per suite per category, random with rng 0, all
difficulty levels), seed 7, init_idx 0, 1 episode per variant, compiled inference. 10 servers (5×H200 GPU2 + 5×H200 GPU3).

| category | n | **strip (default)** | raw `task.language` | published FastWAM |
|---|---|---|---|---|
| Camera Viewpoints | 30 | **46.7%** | 16.7% | 48.4 |
| Robot Initial States | 30 | **76.7%** | 33.3% | 75.7 |
| Language Instructions | 30 | **63.3%** | 63.3% | 66.8 |
| Light Conditions | 30 | **90.0%** | 76.7% | 96.6 |
| Background Textures | 30 | **63.3%** | 50.0% | 68.5 |
| Sensor Noise | 30 | **60.0%** | 33.3% | 70.1 |
| Objects Layout | 30 | **90.0%** | 66.7% | 76.5 |
| **TOTAL** | 210 | **70.0%** | 48.6% | 70.8 |

| suite | n | strip | raw |
|---|---|---|---|
| libero_spatial | 52 | 69.2% | 61.5% |
| libero_object | 53 | 79.2% | 66.0% |
| libero_goal | 53 | 73.6% | 26.4% |
| libero_10 | 52 | 57.7% | 40.4% |

n=30 per category → ±~9 pp (1 s.e.); the strip run matches the published total within noise. The raw run shows that
the variant suffix in `task.language` ("… view 0 0 100 0 0 initstate 80", "… light 39") is a large language
perturbation for FastWAM (umT5 conditioning), especially on libero_goal (several tasks per scene).
Results: `runs/baseline210_strip/` (default) and `runs/baseline210/` (raw).

**Wall time per episode** (one process, 5 processes sharing an H200, 320-core host under load ~50):
median 12.6 s, mean 23.4 s (success episodes are shorter; a 400-step failure ≈ 15 s, a 700-step libero_10 failure ≈ 25-30 s).
Sensor-Noise variants are CPU-bound by LIBERO-Plus' per-frame image corruption (ImageMagick/scikit-image):
median 50 s, worst case 330-830 s. Per-category medians (s): Background 17.4, Camera 15.5, Language 10.5, Light 10.9,
Layout 10.3, Robot 9.1, Noise 49.8. Throughput: 210 episodes in 12.8 min on 8 servers (≈ 16 episodes/min); 10 servers ≈ 20/min.
**GPU memory per process:** 25.4 GB (max reserved), constant. **Processes per GPU:** 5 per H200 (127 GB of 141 GB),
3 per RTX Pro 6000 96 GB. GPU utilization is low (inference is ~0.1 s per query); the simulator is the bottleneck,
so more processes per GPU only help while CPU is not saturated.

## Validation

`python validate.py` (results in `runs/validate/validation.json`), 5 episodes = `samples/determinism5.jsonl` (one each of
Background/Camera/Language/Layout/Noise):
* **Determinism:** the same 5 episodes run twice → **5/5 identical** (success, steps and bitwise-identical executed
  action trace `action_sha1`), both within one process and across processes on a different GPU (GPU 2 vs GPU 3): 5/5.
* **Override exactness:** identity override (2 tensors overwritten with their own values, `arms=[orig,edit]`) → edit
  trajectory identical to orig 5/5; `restore_verified` true. Destructive overrides (`action head = 0`; Gaussian noise
  0.5σ on `action_expert.blocks.20.ffn.2.weight`) change all 5 trajectories (success 0/5 and 1/5 vs 2/5 orig);
  an orig run after these edits is bitwise identical to the pre-edit run 5/5 → in-place edits work under CUDA graphs and
  are fully undone.
* **Eager vs compiled:** recording (eager) gives slightly different numerics than compiled CUDA graphs (action traces differ, 2/2
  success outcomes equal) → compare arms within one mode.
* **save_frames + probe:** probing the saved step-0 inputs reproduces the executed rollout chunk exactly;
  probe with an edit returns different actions; attention maps `[24, 32, 98]`, ~93% of action-query mass on video tokens (layer 15).
* Activation files: 42 arrays for 3 action + 2 video blocks × {attn_out, mlp_out, resid}; action [1024], video agent/wrist [3072].
