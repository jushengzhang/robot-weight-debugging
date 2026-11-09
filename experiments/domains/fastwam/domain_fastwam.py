"""Weight-space programming domain: FastWAM (world-action model) on LIBERO-Plus.

Env vars: WSP_TARGET (perturbation category the run should repair, e.g. "Camera Viewpoints").
Paths are absolute (the kernel has a cleared environment but no filesystem sandbox on this host).
"""
import hashlib, importlib.util, json, os, sys

WSP = "/mnt/SSD2_16TB/jusheng/wsp"
KERNEL_PYTHON = WSP + "/envs/kernel/bin/python"
CKPT = WSP + "/fastwam/weights/fastwam_libero_uncond.safetensors"
BENCH = WSP + "/bench"
READY = WSP + "/fastwam/BACKEND_READY"
TARGET = os.environ.get("WSP_TARGET", "Camera Viewpoints")
LIBERO_CLS = ("/mnt/SSD2_16TB/jusheng/Documents/Codex/2026-10-02/vla-wam-cvpr-gemini-api-key/work/baselines/"
              "LIBERO-plus/libero/libero/benchmark/task_classification.json")

TASK = """TASK. You are given the weights of FastWAM, a world-action model that controls a Franka robot arm in the LIBERO simulator (a released checkpoint, fine-tuned on the LIBERO demonstrations). Your job is to make it more ROBUST to one kind of visual/semantic perturbation, "{target}", by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when the perturbation makes it fail, identify the components responsible, and repair them with targeted changes, without hurting its performance on the other perturbation categories.

THE MODEL. FastWAM is a mixture-of-transformers (MoT) diffusion model with two experts that share self-attention in every layer:
- the VIDEO expert (a Wan2.2-TI2V-5B video DiT: 30 blocks, hidden 3072, 24 heads of 128, FFN 14336, 48-channel VAE latents, patch 1x2x2) processes the clean latent of the current observation (two cameras, agent view and wrist view, 224 px each, tiled into one image and encoded by the frozen Wan2.2 VAE, which is not in the editable weights);
- the ACTION expert (an ActionDiT: 30 blocks, hidden 1024, 24 heads of 128, FFN 4096) denoises an action chunk (a horizon of future 7-D end-effector delta actions plus gripper) with flow matching (10 Euler steps from Gaussian noise at test time; the video tokens stay at timestep 0, i.e. FastWAM does NOT imagine future frames at test time, it only uses the video expert as an observation encoder whose keys/values the action tokens attend to).
In each of the 30 layers, queries/keys/values of video and action tokens are computed by each expert's own projections and attended jointly (video tokens attend causally to the first frame; action tokens attend to the video tokens and to each other); each expert has its own cross-attention to the text-instruction context (frozen umt5-xxl embeddings, with the robot proprioception appended as an extra context token through a small proprio encoder), its own AdaLN timestep modulation and its own MLP. The action chunk is executed open loop for 10 steps, then the model is queried again with the new observation.
Tensor names and shapes are in NAMES / SHAPES (see below); "video_expert.*" / "action_expert.*" (or similar prefixes; inspect NAMES) are the two experts.

THE BENCHMARK. LIBERO-Plus: the 4 LIBERO suites (spatial, object, goal, long/libero_10) with 7 perturbation categories (Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise, Objects Layout) and difficulty levels 1-5. An episode succeeds if the task's goal predicate holds within the step limit (400 steps; 700 for libero_10). Reported (not verified by us) success of this checkpoint: ~71% overall, weakest on Camera Viewpoints (~48%), Background (~69%), Language (~67%); Light is ~97%.

HOW IT WILL BE MEASURED. After you finish, the edited and the unedited model are evaluated on HELD-OUT LIBERO-Plus variants you never see (a fixed half of all variants, chosen by a hash), with fixed seeds: success rate on "{target}" (the goal: raise it) and on every other category (must not drop; a drop of more than ~2 points on the unperturbed-like categories counts against you). Only weight values change: same tensor names and shapes, same inference code and settings.

WHAT COUNTS. Edits that come from understanding the network: find what goes wrong when the perturbation causes failure (e.g. which video-token features shift under the perturbation and which action-expert heads read them), localize the responsible components causally, and change them in a way you can justify with evidence. Out of scope: training of any kind; edits that only multiply a whole tensor by a constant; global changes to the noise schedule or action scaling that are not tied to a mechanism.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU; CPU with many cores): variables survive between code blocks, like a notebook. Reply with exactly one block that starts with ```python to run code; if its last line is an expression its value is shown. One execution may take up to 900 s and 96 GB; output is cut at 10,000 characters; at most {max_python} code runs. Predefined: NAMES (all editable tensor names), SHAPES, load(name) (float32 torch tensor: your current edit if any, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() (names you edited), WORK (your read-write folder, kept for the whole run), and BENCH (read-only): BENCH/practice_variants.json lists the PRACTICE variants (suite, id, name, category, difficulty) - the only ones you may run; the held-out ones are not listed. Gradients are disabled.
2. A forward-pass tool with a budget of {passes} passes. Reply with exactly one block that starts with ```forward and contains JSON. Each request costs one pass:
 (a) ROLLOUT: {{"type": "rollout", "model": "original" | "edited" | "both", "episodes": [{{"suite": "...", "id": <practice variant id>, "seed": <int>}}, ...] OR "sample": {{"category": "...", "suite": optional, "difficulty": optional list, "n": <int>, "seed": <int>}}, "record": optional {{"action_layers": [...], "video_layers": [...], "what": "attn_out" | "mlp_out" | "resid"}}, "save_frames": optional bool}}. At most 256 episodes per pass ("both": at most 128 per model, run on the same variants and seeds, returning the paired difference with its standard error). Returns success (overall, by category, suite and difficulty), steps, and failure modes; per-episode records in WORK/roll_<k>.jsonl; with "record", per-episode activation summaries (mean over the episode's policy queries of the requested module outputs, video tokens averaged per camera, action tokens averaged) as .npy in WORK/roll_<k>_acts/ with index.json; with "save_frames", the first observation and every 50th observation of each episode as .png in WORK/roll_<k>_frames/ and the exact policy inputs needed for probes.
 (b) PROBE (cheap, no simulation): {{"type": "probe", "model": "original" | "edited", "inputs": "<path of saved policy inputs from a rollout with save_frames, or a list of them>", "return": {{"actions": true, "layers": [...], "what": "attn_out" | "mlp_out" | "resid" | "attn_maps", "noise_seed": <int>}}}}: runs the policy on stored observations (up to 512 inputs) and returns the predicted action chunks and requested activations as .npy in WORK/probe_<k>/ (attention maps: at most 4 inputs, at most 4 layers).
The forward tool may not be online for the first part of the run (the simulator backend is being set up); a request made while it is offline is refused WITHOUT costing a pass - use that time to study the weights. A rollout pass of 256 episodes takes on the order of 10-30 minutes.

HOW TO WORK.
1. Diagnose first: compare rollouts on "{target}" variants with less perturbed ones of the same tasks; look at how it fails (saved frames, action statistics), and at which internal features shift under the perturbation.
2. Localize causally: correlation between a component and success is not enough; show that changing it changes the behaviour as predicted, with a dose-response.
3. Measure with power: success over n episodes has a standard error of about 50/sqrt(n) points. Compare original and edited on the SAME variants and seeds ("model": "both"), and keep an edit only if its gain is clearly larger than its standard error on variants not used to choose it. Check other categories for damage before you finish.
4. Plan your passes; keep WORK/notes.md up to date (findings with numbers, edits tried and their measured effect, current best edit; it is shown to you again if the conversation is restarted); keep copies of good edits (e.g. copy WORK/edited to WORK/best_1/).
5. Print compact summaries, never whole arrays; outputs are ASCII only.

RULES. Not allowed: (a) gradient descent or any training (no backward passes, autograd, optimizers, or iterative fitting of weights to a loss); (b) running the network yourself in the kernel on inputs (only the forward tool runs it); (c) reading benchmark files other than BENCH/practice_variants.json (in particular not the held-out variant list) or anything outside WORK, BENCH and the weights; (d) internet access. Allowed: any analysis of weights and forward-tool results, any direct edit of weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {passes} forward passes are used; then make your final choice (the tensors in WORK/edited are submitted) and reply without a code block, saying what you changed, the evidence, and why you expect it to help."""

PRELUDE = r'''
import json, os, numpy as np, torch
torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
from safetensors import safe_open
CKPT = "__CKPT__"; WORK = "__WORK__"; BENCH = "__BENCH__"
_f = safe_open(CKPT, "pt")
NAMES = sorted(_f.keys())
SHAPES = {n: tuple(_f.get_slice(n).get_shape()) for n in NAMES}
_ED = WORK + "/edited"
def load_original(name):
    return _f.get_tensor(name).float()
def load(name):
    p = _ED + "/" + name + ".npy"
    return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)
def save_edit(name, t):
    assert name in SHAPES, "unknown tensor " + name
    a = (t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32))
    assert a.shape == SHAPES[name], (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), "non-finite values"
    o = load_original(name).numpy()
    if a.size > 1 and np.abs(o).max() > 0:
        r = a.reshape(-1) / np.where(o.reshape(-1) == 0, np.nan, o.reshape(-1))
        r = r[np.isfinite(r)]
        assert not (r.size and np.allclose(r, r[0], rtol=1e-4) and not np.allclose(r[0], 1.0)), "refused: whole-tensor scaling"
    os.makedirs(_ED, exist_ok=True); np.save(_ED + "/" + name + ".npy", a.astype(np.float32))
    return "saved " + name
def drop_edit(name):
    p = _ED + "/" + name + ".npy"
    if os.path.exists(p): os.remove(p)
def edited():
    return sorted(f[:-4] for f in os.listdir(_ED)) if os.path.isdir(_ED) else []
'''


def _split():
    """Practice = variants whose hash is even; held-out = odd. Deterministic."""
    os.makedirs(BENCH, exist_ok=True)
    pp, hp = BENCH + "/practice_variants.json", WSP + "/fastwam/heldout_variants.json"
    if os.path.exists(pp) and os.path.exists(hp):
        return
    d = json.load(open(LIBERO_CLS))
    prac, held = [], []
    for suite, items in d.items():
        for it in items:
            rec = {"suite": suite, "id": it["id"], "name": it["name"], "category": it["category"],
                   "difficulty": it.get("difficulty_level")}
            h = int(hashlib.sha256(f"{suite}:{it['id']}".encode()).hexdigest(), 16)
            (prac if h % 2 == 0 else held).append(rec)
    json.dump(prac, open(pp, "w")); json.dump(held, open(hp, "w"))
    os.chmod(hp, 0o600)


def setup(R):
    global PRELUDE
    _split()
    PRELUDE = PRELUDE.replace("__CKPT__", CKPT).replace("__WORK__", R + "/work").replace("__BENCH__", BENCH)


def forward(req, R, k):
    if not os.path.exists(READY):
        raise ValueError("the forward-pass tool is not online yet (the simulator backend is still being set up); "
                         "this request did not count. Continue with weight analysis and try again later")
    spec = importlib.util.spec_from_file_location("fwd", WSP + "/harness/fastwam_forward.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.forward(req, R, k, target=TARGET)
