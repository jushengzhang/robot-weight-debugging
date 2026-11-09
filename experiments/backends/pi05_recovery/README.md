# π0.5 (openpi pi05_libero) × LIBERO-Recovery (host js186)

LIBERO-Recovery (arXiv 2609.05178, github.com/liulin815/LIBERO-Recovery @832d54d, data: ModelScope `ataier/*`) is a
failure-recovery benchmark: 4,803 failure scenes = (.bddl + recorded MuJoCo flat sim state), dumped from real rollouts
of starVLA Qwen3-VL-PI policies and VLM-labelled with recovery level L1 retry / L2 adapt / L3 restore object /
L4 remove an obstruction (465 L4 constructed by injection). Success = original LIBERO task goal (env `done`) within 520 steps.

Reuses (read-only) the wsp_pi05 venv, checkpoint, caches and LIBERO-orig. Nothing in wsp_pi05 is modified.

| path | what |
|---|---|
| `recover_server.py` | COPY of `wsp_pi05/policy_server.py` (sha256 in `COPIED_FROM.sha256`) + scene episodes (episode dict has `bddl`), CPU fallback (`WSP_DEVICE=cpu`), placement fallback, BDDL `:objects` normalizer. Same socket protocol / weight overrides / record / probe as wsp_pi05 |
| `client.py` | copy of wsp_pi05 client (only `call` is used) |
| `run_bench.py` | work-queue driver; re-scans `run/sock/*.sock` every 30 s (servers can be added live); resume-safe `results.jsonl`; `--override DIR --arms orig,edit` for weight edits |
| `start_servers.sh` / `stop_servers.sh` | `bash start_servers.sh "6:1 7:1"` (GPU 6/7 only, refuses if <9 GB free); `cpu:N` = CPU inference servers (EGL on GPU 6); stop kills only pids in `run/pids` |
| `make_sample.py` | stratified suite × level sample (libero_90 excluded: pi05_libero is not trained on it) |
| `fix_scenes.py` | repairs two released-data BDDL bugs into `data/scenes_fixed/` (see below) |
| `analyze.py` | tables by level / suite / gripper / stage + failure-mode heuristics |
| `data/FailureScenes` | ModelScope `ataier/LIBERO_Recovery_FailureScenes` (95 MB, 4,803 scenes, annotations, index) |
| `data/Assets/scenes` | ModelScope `ataier/LIBERO_Recovery_Assets` curated 880-scene set (launcher default upstream) |
| `third_party/LIBERO-Recovery` | upstream repo clone (reference only; its LIBERO `envs/` is identical to LIBERO-orig) |
| `samples/rec220x2_fixed.jsonl` | baseline sample: 220 scenes (≤15 per suite×level) × {open, close} gripper start |
| `samples/hard_both_fail.jsonl` | the 59 scenes (118 eps) where π0.5 failed both trials |
| `runs/baseline_rec220x2/` | baseline results (`results.jsonl`, `analysis.md`) |

## Episode protocol (= upstream `eval_libero_custom_scene.py`, with π0.5 I/O from wsp_pi05)
env from scene .bddl → `set_init_state(recorded state + uniform ±2 cm xy offset on every free-joint object)` → OSC goal
re-anchored → gripper driven closed/open for 20 zero-arm steps → ≤520 policy steps (no settle wait), chunk 10 / replan 5,
openpi preprocessing (180° rotate, resize_with_pad 224). Upstream runs 5 close + 5 open trials per scene; we ran 1 + 1.
Episode fields: `success, success_at_t0` (goal already satisfied before the policy acts), `steps, grip_flips,
frac_grip_closed, eef_path_m, eef_net_m, obj_deltas, placement_relaxed, bddl_normalized, level, stage, grip`.

Deliberate deviations / upstream bugs:
* **Instruction** = LIBERO task language (annotation `task`), what pi05_libero was trained on. Upstream feeds the BDDL
  `(:language …)` line, a paraphrase for many tasks ("pick the akita black bowl …"). Pass no `instruction` to get upstream behaviour.
* **Perturbation** offset: the upstream script indexes the flat state `[time, qpos, qvel]` with raw qpos addresses, so it
  perturbs (qpos[adr-1], x) = (previous object's quat-z or a finger joint, x) instead of (x, y). We perturb x, y
  (`pert_mode: "ref"` reproduces upstream).
* **Released-data BDDL bugs** (would crash upstream too): 975/1,139 libero_object scenes use `(:target main_table)` in a
  floor arena (KeyError main_table → `floor`); 57/562 libero_goal scenes drop goal-only regions (e.g. stove_front_region →
  copied from the original LIBERO bddl); some scenes declare one category on several lines (merged, as upstream `fix_bddl_objects.py`).
* Tight 2 cm init regions often make LIBERO's (throw-away) placement sampler fail → retried without the overlap check.
* Per-episode RNG = hash(scene, seed, trial) (shardable) instead of one sequential stream.

## Run
```bash
cd /mnt/SSD_16TB/jusheng/wsp_recover
bash start_servers.sh "6:1 7:1"          # ~8 GB each; ready = run/sock/r6_0.ready
P=/mnt/SSD_16TB/jusheng/wsp_pi05/venv/bin/python
$P run_bench.py --episodes samples/hard_both_fail.jsonl --out runs/x [--override DIR --arms orig,edit]
$P analyze.py runs/x
bash stop_servers.sh
```
GPU server: ~36 s for a 520-step failure, ~5-25 s for a success. CPU server (24 threads): ~8 min per 520-step failure.
