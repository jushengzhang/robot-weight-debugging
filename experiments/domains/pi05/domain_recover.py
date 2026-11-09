"""Weight-space programming domain: pi0.5 (openpi pi05_libero) on LIBERO-Recovery failure scenes (host js186).
Same model, kernel and weights as domain_pi05; episodes start from recorded failure states."""
import importlib.util, os

H = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("domain_pi05_base", os.path.join(H, "domain_pi05.py"))
_b = importlib.util.module_from_spec(_s); _s.loader.exec_module(_b)
KERNEL_PYTHON, CKPT = _b.KERNEL_PYTHON, _b.CKPT
BENCH = "/mnt/SSD_16TB/jusheng/wsp_recover/wsp/bench"
TARGET = "failure recovery"


def _sec(a, b):
    t = _b.TASK
    return t[t.index(a):t.index(b)]


INTRO = """TASK. You are given the weights of pi0.5, a vision-language-action model (VLA) that controls a Franka robot arm in the LIBERO simulator (the released openpi checkpoint "pi05_libero", fine-tuned on the LIBERO demonstrations; it succeeds on ~96% of the original LIBERO tasks). Your job is to make it better at RECOVERING FROM FAILURES, by EDITING ITS WEIGHTS DIRECTLY. Each episode of the LIBERO-Recovery benchmark starts in the middle of a task, in a recorded state where a robot policy had already gone wrong (the grasp missed, the object was dropped or knocked over, an obstruction is in the way...). From there pi0.5 usually has every skill it needs to finish the task, yet it fails 40% of the time, and in about 80% of its failures it gets stuck in a RE-GRASP LOOP: it closes and opens the gripper again and again (6 or more flips) around the same place until the time limit, instead of adapting (e.g. moving to where the object now is). Find out what goes wrong inside the network in these episodes, identify the components responsible, and repair them with targeted changes, without hurting its ordinary task performance.

"""

BENCHMARK = """THE BENCHMARK. LIBERO-Recovery (2026): failure scenes from the 4 LIBERO suites (spatial, object, goal, libero_10), each a recorded simulator state plus the original task instruction, labelled by recovery level: L1 retry (redo the failed grasp), L2 adapt (the situation changed; adjust the approach), L3 restore (an object was displaced/dropped and must be recovered first), L4 obstruction (something blocks the way). Every free object is shifted by a random +/-2 cm at the start (depends on the seed), and the gripper starts either open or closed ("grip"). An episode succeeds if the task's goal predicate holds within 520 policy steps (a few scenes are already solved at the start; they are reported separately). Baseline of this checkpoint on 440 episodes: 59.5% (L1 60, L2 72, L3 54, L4 53; libero_10 L3 17%); failures: 141/178 re-grasp loops, 27 wandering, 7 stalls.

HOW IT WILL BE MEASURED. After you finish, the edited and the unedited model are evaluated on HELD-OUT failure scenes you never see (a fixed half of all scenes, chosen by a hash), both gripper starts, fixed seeds: success rate (the goal: raise it) overall and per level, and the number of re-grasp-loop failures. Separately, ordinary performance on the original LIBERO tasks must not drop by more than ~2 points. Only weight values change: same tensor names and shapes, same inference code and settings.

WHAT COUNTS. Edits that come from understanding the network: find what drives the loop (e.g. which observation features or attention patterns keep the policy re-issuing the same grasp, whether the model "sees" that the object moved, how the gripper command is produced), localize the responsible components causally, and change them in a way you can justify with evidence. Out of scope: training of any kind; edits that only multiply a whole tensor by a constant; global changes to the noise or action scaling that are not tied to a mechanism.

"""

FWD = """2. A forward-pass tool with a budget of {passes} passes. Reply with exactly one block that starts with ```forward and contains JSON. Each request costs one pass:
 (a) ROLLOUT: {{"type": "rollout", "model": "original" | "edited" | "both", "episodes": [{{"id": <practice scene id>, "grip": "open" | "close", "seed": <int>}}, ...] OR "sample": {{"suite": optional, "level": optional ("L1".."L4" or a list), "stage": optional, "n": <number of scenes>, "seed": <int>, "grips": optional list (default both)}}, "record": optional {{"vlm_layers": [...], "expert_layers": [...], "vision_layers": [...], "what": "attn_out" | "mlp_out" | "resid"}}, "save_frames": optional bool}}. At most 256 episodes per pass ("both": at most 128 per model, run on the same scenes, grips and seeds, returning the paired difference with its standard error). Returns success (overall, by level, suite, stage and grip), re-grasp-loop counts, gripper-flip statistics; per-episode records in WORK/roll_<k>.jsonl (success, steps, grip_flips, frac_grip_closed, eef_path_m, eef_net_m, obj_deltas, path of the executed actions .npy); with "record", per-episode activation summaries (mean over the episode's policy queries; VLM outputs averaged separately over base-image, wrist-image and instruction tokens, expert outputs over the action tokens) as .npy in WORK/roll_<k>_acts/ with index.json; with "save_frames", the first observation and every 50th step as .png in WORK/roll_<k>_frames/ and the exact policy inputs needed for probes.
 (b) PROBE (cheap, no simulation): {{"type": "probe", "model": "original" | "edited" | "both", "inputs": "<path of saved policy inputs from a rollout with save_frames, or a list of them>", "return": {{"layers": [expert layers], "vlm_layers": [...], "vision_layers": [...], "what": "attn_out" | "mlp_out" | "resid" | "attn_maps"}}}}: runs the policy on stored observations (up to 512 inputs) and returns the predicted action chunk and requested activations (per call) as .npy in WORK/probe_<k>/ (attention maps of the action expert over the prefix: at most 4 inputs, at most 4 layers).
Same scene, grip and seed give the same episode (deterministic). A rollout pass of 256 episodes takes on the order of 10 minutes.

"""

WORK = """HOW TO WORK.
1. Diagnose first: compare looping failures with successful recoveries (and with ordinary episodes); look at the saved frames and action traces (where does it grasp, does it ever move toward the displaced object), and at which internal features or attention patterns differ.
2. Localize causally: correlation between a component and success is not enough; show that changing it changes the behaviour as predicted, with a dose-response.
3. Measure with power: success over n episodes has a standard error of about 50/sqrt(n) points. Compare original and edited on the SAME scenes, grips and seeds ("model": "both"), and keep an edit only if its gain is clearly larger than its standard error on scenes not used to choose it. Check for damage on scenes it already solves before you finish.
4. Plan your passes; keep WORK/notes.md up to date (findings with numbers, edits tried and their measured effect, current best edit; it is shown to you again if the conversation is restarted); keep copies of good edits (e.g. copy WORK/edited to WORK/best_1/).
5. Print compact summaries, never whole arrays; outputs are ASCII only.

"""

_have = _sec("WHAT YOU HAVE.", "2. A forward-pass tool")
_have = _have.replace("BENCH/practice_variants.json lists the PRACTICE variants (suite, id, name, category, difficulty) - the only ones you may run; the held-out ones are not listed.",
                      "BENCH/practice_scenes.json lists the PRACTICE failure scenes (id, scene_id, suite, level, stage, instruction, l4_injected) - the only ones you may run; the held-out ones are not listed.")
_have = _have.replace("BENCH/paligemma_tokenizer.model", "BENCH/paligemma_tokenizer.model")
_rules = _sec("RULES.", "HOW THE SESSION RUNS.").replace("BENCH/practice_variants.json", "BENCH/practice_scenes.json") \
    .replace("in particular not the held-out variant list", "in particular not the held-out scene list")
TASK = (INTRO + _sec("THE MODEL", "THE BENCHMARK.") + BENCHMARK + _have + FWD + WORK + _rules
        + _b.TASK[_b.TASK.index("HOW THE SESSION RUNS."):])
PRELUDE = _b.PRELUDE


def setup(R):
    global PRELUDE
    assert os.path.exists(BENCH + "/practice_scenes.json")
    PRELUDE = _b.PRELUDE.replace("__CKPT__", CKPT).replace("__WORK__", R + "/work").replace("__BENCH__", BENCH)


def forward(req, R, k):
    spec = importlib.util.spec_from_file_location("fwd", os.path.join(H, "recover_forward.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.forward(req, R, k, target=TARGET)
