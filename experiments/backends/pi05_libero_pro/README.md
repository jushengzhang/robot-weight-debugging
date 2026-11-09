# π0.5 (openpi `pi05_libero`, PyTorch) × LIBERO-PRO backend (host js186)

LIBERO-PRO (arXiv 2510.03827, github.com/Zxy-MLlab/LIBERO-PRO, HF `zhouxueyang/LIBERO-Pro`) is a third benchmark for Weight-Space Programming.
It follows the `wsp_recover` pattern:
* the server is a copy of `wsp_pi05/policy_server.py` with LIBERO-PRO episodes added;
* the `wsp_pi05` venv, checkpoint and caches are reused read-only;
* nothing in `wsp_pi05` / `wsp_recover` is modified.

The twin FastWAM backend is `120:/mnt/SSD2_16TB/jusheng/wsp/libero_pro`. Both use the same LIBERO-PRO copy, episode list, split and baseline sample; the files were generated on 120 and rsynced.

| path | what |
|---|---|
| `pro_server.py` | COPY of `wsp_pi05/policy_server.py` (incl. its CPU-side override cast) + LIBERO-PRO episodes + CPU fallback (`WSP_DEVICE=cpu`, as wsp_recover). Built by `patch_server.py` from `policy_server.orig.py` (sha256 in `COPIED_FROM.sha256`) |
| `client.py` | copy of `wsp_pi05/client.py` (key-based resume, PRO summary; built by `patch_client.py`) |
| `run_bench.py` | work-queue driver (re-scans `run/sock` every 30 s, resume-safe), `--override DIR --arms orig,edit`, `--language pro|filename|raw` |
| `start_servers.sh` / `stop_servers.sh` | `bash start_servers.sh "6:1"` (GPU 6/7 only, refuses unless ≥10 GiB free per server); `cpu:N` = CPU-inference servers `c<egl gpu>_k` (EGL rendering on `CPU_EGL_GPU`, default 6; `CPU_THREADS` default 16). Stop kills only the pids in `run/pids` (cmdline-verified) |
| `gpu_watch.sh` | starts ONE GPU server (`p6_0` or `p7_0`) once GPU 6 or 7 has ≥20 GB free; `run_bench.py` attaches it automatically. `touch run/stop_watch` stops it |
| `validate_pro.py` | smoke/validation (`runs/validate/validation.json`) |
| `make_episodes.py`, `make_sample.py`, `analyze.py`, `gen_env.py`, `hfdl.py` | episode list, stratified sample, tables/failure modes, env-perturbation generator, HF downloader |
| `episodes/` | `all_episodes.jsonl`, `practice.jsonl`, `heldout.jsonl` (600), `orig_episodes.jsonl` |
| `samples/baseline260.jsonl` | baseline sample (practice split; identical on 120) |
| `third_party/LIBERO-PRO`, `data/LIBERO-Pro-hf`, `libero_cfg/` | benchmark copy, raw HF data, LIBERO config (`LIBERO_CONFIG_PATH`; `~/.libero` untouched) |
| `overrides/` | copies of `wsp_pi05/overrides/test_*` (validation) |

## Versions
* venv `wsp_pi05/venv` (Python 3.11): torch 2.12.0+cu130, transformers 4.53.2 (openpi patch), mujoco 3.3.2, robosuite 1.4.1.
* openpi @ 215abfb. Checkpoint `wsp_pi05/ckpt/pi05_libero_pytorch` (converted from `gs://openpi-assets/checkpoints/pi05_libero`).
* `PYTHONPATH=third_party/LIBERO-PRO` puts LIBERO-PRO ahead of the venv's editable LIBERO-plus. The server `info` reports `libero` = the LIBERO-PRO copy.
## LIBERO-PRO: what was installed

* **Code:** `third_party/LIBERO-PRO` = `github.com/Zxy-MLlab/LIBERO-PRO` master @ `eafdb809426b` (2026-03-23). This is a separate copy; no existing backend was modified.
  Three local changes, all additive:
  1. An empty `libero/__init__.py`. Upstream ships `libero/` as a namespace package, so the venvs' LIBERO-plus regular package shadowed it (`import libero.libero` resolved to LIBERO-plus).
  2. The HF data (`bddl_files/` and `init_files/` for `<suite>_{lan,object,swap,task}`) copied in, as the upstream README says to.
  3. The generated `<suite>_env` BDDL and init files (see below) plus `bddl_files/gen_env_manifest.json`.
  `git status` lists only these untracked paths.
* **Data:** `data/LIBERO-Pro-hf` = HF dataset `zhouxueyang/LIBERO-Pro` @ `c86fc3b8293185a6f373677018ff3e37f8391602` (2026-07-22), 676 files, 12 MB. All 353 entries in `SHA256SUMS.txt` are OK. Downloaded with `hfdl.py`.
  The original suites' init files on HF are byte-identical to the repo's.
* **Environment perturbation (not shipped on HF):** upstream creates it on the fly (`perturbation.create_env`). `gen_env.py` does the same thing reproducibly:
  * BDDL: upstream `EnvironmentReplacePerturbator`. The released code **hardcodes `new_env = "living_room_table"`**, so the 5 libero_10 `LIVING_ROOM_*` tasks come out identical to the original (`env_identical_to_orig: true` in the episode list). The other 35 tasks move to the living-room table.
  * Init states: the upstream `notebooks/generate_init_states.py` procedure (50 fresh `OffScreenRenderEnv` builds at 128x128, `get_sim_state`, the same zip/pickle `.pruned_init`), but seeded per (task, i) so the files are reproducible. Generated on 120 and rsynced, so both hosts use identical files.
* **Seven-case "robustness extension" NOT included.** HF also ships `bddl_files/0{1..7}_*` (40 tasks × 7 cases: glare/noise, camera angle, runtime object move, texture, occlusion, shape, initial pose).
  Its runtime (`:perturbation_config` parser, observation transforms, runtime object mover) exists only on the unmerged upstream branch `origin/codex/pr1-bddl-robustness` (2b910b5, 2026-07-17). It is not in the paper and has no published numbers.
  The episode list covers the paper's 5 perturbation types only.

## Episodes (`episodes/`)
`make_episodes.py <LIBERO-PRO> episodes/`. One episode = (suite, perturbation type, task, init_idx 0..49), seed 7.
Key = `<suite>/<ptype>/<task>/<init:02d>`.

| ptype (paper name) | LIBERO-PRO suite suffix | episodes | practice | what changes |
|---|---|---|---|---|
| object | `_object` | 2000 | 998 | object appearance/size swapped (e.g. `akita_black_bowl` → `bigger_akita_black_bowl`, `plate` → `yellow_plate`); instruction unchanged |
| position | `_swap` | 2000 | 976 | objects' init regions swapped (new layouts); instruction unchanged |
| semantic | `_lan` | 2000 | 991 | paraphrased instruction ("place the bowl on top of the plate") |
| task | `_task` | 2000 | 989 | new goal and new instruction in the same scene ("put the wine bottle on the plate") |
| environment | `_env` | 2000 | 989 | table/scene replaced by `living_room_table` (5 libero_10 tasks unchanged, see above) |
| **total** | | **10000** | **4943** (held-out 5057) | 4 suites (spatial/object/goal/10) × 10 tasks × 50 inits each |

* `all_episodes.jsonl` (10000), `practice.jsonl` (4943), `heldout.jsonl` (5057, **chmod 600**). The split is `int(sha256(key), 16) % 2 == 0` → practice.
* `orig_episodes.jsonl` (2000) holds the unperturbed suites from the same copy. It is a reference only and is not in the split.
* Fields: `key, suite, ptype, pro_suite, task, task_idx, init_idx, seed, max_steps` (LIBERO-PRO/openpi 220/280/300/520), `instruction` (what the servers feed by default), `instruction_bddl, instruction_orig, instruction_changed, bddl, bddl_sha256, detail` (per-type diff: swapped regions, replaced objects, new goal), and `env_identical_to_orig` for environment episodes.
* **Instruction (default `language="pro"`):**
  * For semantic/task, the model gets the perturbed BDDL `(:language)` (lowercased).
  * For all other types it gets the standard LIBERO instruction `task.language`, which is what both models were trained on.
  * Reason: for 25/40 tasks the *original* BDDL `(:language)` line differs from the training instruction ("pick the akita black bowl …" vs "pick up the black bowl …"), so feeding the raw BDDL text to every episode would add a language shift on top of the intended perturbation.
  * `language="filename"` feeds `task.language` everywhere. That is what the upstream OpenVLA `run_libero_eval` snippet literally does (`task.language` comes from the file name, which LIBERO-PRO keeps unchanged for `_lan`/`_task`), which would make semantic a no-op and task an instruction/goal mismatch. The paper's semantic ≈ original numbers are consistent with this, but we cannot tell.
  * `language="raw"` feeds the BDDL text verbatim.

## Server (`pro_server.py`)
* A COPY of the backend's `policy_server.py` (sha256 in `COPIED_FROM.sha256`; the pristine copy is `policy_server.orig.py`), built reproducibly by `patch_server.py`. Every edit is a checked single-occurrence string replacement.
* Protocol, request fields, weight overrides, restore verification, `record`, `save_frames` and `probe` are unchanged.
* An episode dict with `"ptype"` is routed to `run_pro_episode`. Old-style LIBERO-Plus episodes still go through the original code path, but the LIBERO-Plus classification is not loaded.
* Extra result fields:
  * `key, ptype, pro_suite, task, instruction*, language_mode, bddl_sha256`;
  * `settle_drift_m` = max xy drift of free objects during the settle steps. Init states spawn objects ~7 cm high, so z is ignored;
  * `done_at_t0`; `obj_moved_m` (3D, from the settled pose); `grip_flips, frac_grip_closed, eef_path_m, eef_net_m`;
  * for task episodes, **`orig_goal_reached` / `orig_goal_step`**: the *original* task's goal predicates, evaluated on the perturbed scene at every replan. This indicates memorized behaviour.
* `variant_id` is set to the key, so the old client code paths keep working.
* Init states are loaded with `torch.load(..., weights_only=False)`. Upstream `get_task_init_states` uses the default, which fails on torch ≥ 2.6 for these numpy pickles.

Common scripts: `client.py` (copy of the backend client; resume key = episode key; PRO summary with the published leaderboard), `run_bench.py` (work queue; re-scans sockets every 30 s, so servers can join a running baseline; resume-safe), `make_sample.py`, `analyze.py` (tables + failure modes), `validate_pro.py` (smoke/validation).

### π0.5 episode protocol
Identical to `wsp_pi05` (openpi `examples/libero/main.py`, which is also LIBERO-PRO's OpenVLA-OFT runner protocol):
* Env: `OffScreenRenderEnv` 256², `env.seed(7)`, reset, `set_init_state(init_states[init_idx])` from the `<suite>_<type>` init file.
* 10 settle steps; chunk 10, replan 5, 10 flow steps; eager mode.
* Max steps 220/280/300/520, which equals LIBERO-PRO's `TASK_MAX_STEPS`.
* Success = env `done` (the perturbed BDDL goal).

## Commands
```bash
cd /mnt/SSD_16TB/jusheng/wsp_pro
PY=/mnt/SSD_16TB/jusheng/wsp_pi05/venv/bin/python
nvidia-smi --query-gpu=index,memory.free --format=csv -i 6,7   # GPU server needs >=10 GiB free (7.8 GB max reserved; wsp_pi05 servers grow)
bash start_servers.sh "6:1"                                     # or: CPU_THREADS=16 bash start_servers.sh "cpu:2"
$PY run_bench.py --episodes samples/baseline260.jsonl --out runs/x [--override DIR --arms orig,edit] [--language pro|filename|raw]
$PY analyze.py runs/x
bash stop_servers.sh
$PY validate_pro.py --socks run/sock/c6_0.sock --record '{"expert_layers":[9],"vlm_layers":[5],"what":["resid"]}'
```
Python API: `import client as C` → `C.rollout(eps, out, socks, weights_override=..., arms=[...])`, `C.probe(...)`, as in `wsp_pi05`.
Throughput:
* CPU server (16-24 threads): ~3 s per query, i.e. a 220-step failure takes ~2.5 min and a 520-step failure ~6 min.
* GPU server: ~0.1 s per query (wsp_pi05 numbers).

## Validation (`runs/validate/validation.json`; CPU server, 20-step episodes)
* Coverage: one episode of each type (original/object/position/semantic/task/environment×2), 0 errors.
* Identity override (`action_out_proj.weight` + `vlm.layers.5.self_attn.o_proj.weight` set to their own values; paired orig/edit): edit == orig 2/2, `restore_verified`.
* `test_zero_head` (action_out_proj = 0) and `test_noise_expert_l10` change 2/2 traces; restore verified.
* orig after the edits is bit-identical 2/2.
* record: activations for vlm.5/expert.9 (calls 4 and 40 for 4 queries), with actions identical to the plain run.
* save_frames + probe: probe(orig) reproduces the saved chunks exactly (2/2); the edit arm differs (max |Δa| 0.003-0.005).
* Earlier runs `runs/validate_v0_staledrift`, `runs/validate_v1_zdrift` differ only in the drift diagnostics (fixed); all override checks passed there too.

## Deviations / notes
* All instruction and env-perturbation choices are in the common section above.
* **The baseline ran on CPU inference** (4 servers × 16 threads; EGL rendering on GPU 6/7, no CUDA memory). GPUs 6/7 had only 6.8 GB free (< 10 GB) for the whole run, so `gpu_watch.sh` never started a GPU server.
  CPU and GPU numerics differ slightly (not bit-identical), so edit experiments must use paired orig/edit arms on the same server type. The success statistics are comparable.
* `settle_drift_m` is measured in xy only, because LIBERO init states spawn objects ~7 cm above the table and they drop during settling.

## Baseline (original weights): `runs/baseline260/` (`results.jsonl`, `summary.md`, `analysis.md`)
* Sample `samples/baseline260.jsonl` (identical on 120), from the **practice** split (`make_sample.py --per-cell 12 --orig-per-suite 5 --seed 0`).
  * 12 episodes per perturbation type × suite (all 10 tasks covered, random init_idx) = 240.
  * Plus 20 unperturbed reference episodes.
* Settings: seed 7, `language=pro`, eager.
* Ran on **4 CPU servers** (see deviations): 19:00-21:55 CST. Wall/episode median 156 s, max 361 s; 12.2 server-hours in total.

Cells are success % (n); brackets give the **published** [π0.5 / π0] from the LIBERO-PRO README leaderboard (repo @eafdb80; arXiv 2510.03827).

| ptype | libero_spatial | libero_object | libero_goal | libero_10 | all | published π0.5 / π0 (mean of 4 suites) | FastWAM (120, same sample) |
|---|---|---|---|---|---|---|---|
| original | 100 (5) | 100 (5) | 100 (5) | 100 (5) | **100** (20) | (orig LIBERO 96.9 openpi) | 95 |
| object | 100 (12) [97/95] | 100 (12) [98/94] | 92 (12) [97/94] | 83 (12) [92/79] | **94** (48) | 96.0 / 90.5 | 83 |
| position | 42 (12) [20/0] | 25 (12) [17/0] | 33 (12) [38/0] | 0 (12) [8/0] | **25** (48) | 20.8 / 0.0 | 2 |
| semantic | 100 (12) [97/97] | 100 (12) [96/90] | 92 (12) [97/93] | 83 (12) [93/82] | **94** (48) | 95.8 / 90.5 | 85 |
| task | 67 (12) [1/0] | 8 (12) [1/0] | 8 (12) [0/0] | 8 (12) [1/0] | **23** (48) | 0.8 / 0.0 | 10 |
| environment | 33 (12) [46/60] | 67 (12) [73/29] | 42 (12) [46/39] | 50 (12) [46/27] | **48** (48) | 52.8 / 38.7 | 27 |
| **all** | 71 (65) | 63 (65) | 57 (65) | 49 (65) | **60.0** (260) | | 45.8 |

The 240 perturbed episodes give **56.7%**. The published π0.5 mean over the 20 cells is 53.2% (π0 44.0%); FastWAM gets 41.7% on the same sample. n=12 per cell, so ±~14 pp (1 s.e.).

**Agreement with the published π0.5 row:**
* object, semantic, position and environment are within ~1 s.e. per type: 94 vs 96, 94 vs 96, 25 vs 21, 48 vs 53.
* Position reproduces the "collapse" the paper reports: 0% on libero_10.
* The one clear disagreement is **task: 23% vs 0.8%**, entirely from libero_spatial (8/12).
  * There the new instruction picks the *other* bowl by a location phrase ("the bowl on the stove", "next to the cookie box", …). These phrases are the referring expressions of *other libero_spatial training tasks*, and π0.5 follows them.
  * The published ~1% would follow if the model never saw the new instruction. With the upstream runner's `task.language` (file name, unchanged by LIBERO-PRO), FastWAM reached the original goal in 44/48 task episodes (see the 120 README). We have not run `language=filename` for π0.5.

**Failure modes** (`analyze.py`; 104 failures, 0 errors, all max-step timeouts):

| ptype | fail | did original task | scene unstable | no object moved | regrasp loop (≥6 flips) | partial |
|---|---|---|---|---|---|---|
| object | 3 | – | 0 | 1 | 2 | 0 |
| position | 36 | – | 1 | 6 | 12 | 17 |
| semantic | 3 | – | 0 | 0 | 3 | 0 |
| task | 37 | **24** | 0 | 0 | 6 | 7 |
| environment | 25 | – | 3 | 5 | 15 | 2 |

* **Task:**
  * The original task's goal was reached in 25/48 episodes (24 failures + 1 that also did the new task). π0.5 executes the memorized task despite the new instruction.
  * Outside libero_spatial it succeeds 3/36.
* **Position:**
  * Half the failures move objects without finishing (wrong object, or placed at the old target location).
  * A third are re-grasp loops (median 5 gripper flips).
  * libero_10 (two-object tasks) is 0/12.
* **Environment:** mostly re-grasp loops on the living-room table (median 13 flips).
  * 5 episodes had >2 cm settle drift (2/5 succeeded); stable scenes: 21/43.
  * The 5 libero_10 living-room tasks are not actually perturbed (released-code hardcode).
* **Object / semantic** (94% each) are near ceiling. The few failures are re-grasp loops on resized/recoloured objects or on libero_10 paraphrases.

## Ready-to-run on GPU (when GPU 6/7 free up)
```bash
cd /mnt/SSD_16TB/jusheng/wsp_pro && PY=/mnt/SSD_16TB/jusheng/wsp_pi05/venv/bin/python
bash start_servers.sh "6:1"     # (>=10 GiB free on GPU 6), more with START=1 ... / "7:1"
$PY run_bench.py --episodes samples/baseline260.jsonl --out runs/baseline260_gpu   # ~260 eps x ~20 s / n_servers
$PY run_bench.py --episodes samples/baseline260.jsonl --out runs/baseline260_filename --language filename   # instruction-protocol check
```
