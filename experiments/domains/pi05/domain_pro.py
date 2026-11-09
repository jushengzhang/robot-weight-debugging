"""Weight-space programming domain: pi0.5 (openpi pi05_libero) on LIBERO-PRO (host js186).
Same model, kernel and weights as domain_pi05; episodes are LIBERO-PRO perturbations of the LIBERO tasks."""
import importlib.util, os

H = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("domain_pi05_base", os.path.join(H, "domain_pi05.py"))
_b = importlib.util.module_from_spec(_s); _s.loader.exec_module(_b)
KERNEL_PYTHON, CKPT = _b.KERNEL_PYTHON, _b.CKPT
BENCH = "/mnt/SSD_16TB/jusheng/wsp_pro/wsp/bench"
TARGET = "LIBERO-PRO perturbations"


def _sec(a, b):
    t = _b.TASK
    return t[t.index(a):t.index(b)]


INTRO = """TASK. You are given the weights of pi0.5, a vision-language-action model (VLA) that controls a Franka robot arm in the LIBERO simulator (the released openpi checkpoint "pi05_libero", fine-tuned on the LIBERO demonstrations; it succeeds on ~96-100% of the original LIBERO tasks). LIBERO-PRO perturbs those tasks, and pi0.5 collapses on some perturbations even though the required skills are the same ones it uses on the original tasks. Your job is to make it more ROBUST to these perturbations by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when a perturbation makes it fail, identify the components responsible, and repair them with targeted changes, without hurting its performance on the original tasks.

"""

BENCHMARK = """THE BENCHMARK. LIBERO-PRO (2025): the 40 tasks of the 4 LIBERO suites (spatial, object, goal, libero_10), each with 50 initial states, under 5 perturbation types ("ptype"): object (object appearance changed), position (objects start in different places), semantic (the instruction is paraphrased), task (a DIFFERENT task in the same scene: new instruction and goal), environment (different table/scene). An episode succeeds if the (perturbed) task's goal holds within the step limit (220 spatial, 280 object, 300 goal, 520 libero_10). Baseline of this checkpoint on 260 practice episodes: original 100%, object 94, semantic 94, position 25, task 23, environment 48. Failure modes: on "task", it mostly performs the ORIGINAL task of that scene instead of the new instruction (in about half of the task failures the original task's goal is reached); on "position", it moves objects without finishing or re-grasps in a loop near where the objects used to be; on "environment", mostly re-grasp loops.

HOW IT WILL BE MEASURED. After you finish, the edited and the unedited model are evaluated on HELD-OUT LIBERO-PRO episodes you never see (a fixed half of all episodes, chosen by a hash), with fixed seeds: success rate on each perturbation type and overall (the goal: raise it), and ordinary performance on the original LIBERO tasks (must not drop by more than ~2 points). Only weight values change: same tensor names and shapes, same inference code and settings.

WHAT COUNTS. Edits that come from understanding the network: find what goes wrong (e.g. why the instruction of a new task is overridden by the scene's habitual task, whether the model localizes moved objects, which observation features or attention patterns drive the failure), localize the responsible components causally, and change them in a way you can justify with evidence. Out of scope: training of any kind; edits that only multiply a whole tensor by a constant; global changes to the noise or action scaling that are not tied to a mechanism.

"""

FWD = """2. A forward-pass tool with a budget of {passes} passes. Reply with exactly one block that starts with ```forward and contains JSON. Each request costs one pass:
 (a) ROLLOUT: {{"type": "rollout", "model": "original" | "edited" | "both", "episodes": [{{"id": <practice episode id>, "seed": <int>}}, ...] OR "sample": {{"ptype": optional (one or a list), "suite": optional, "task": optional, "n": <int>, "seed": <int>}}, "record": optional {{"vlm_layers": [...], "expert_layers": [...], "vision_layers": [...], "what": "attn_out" | "mlp_out" | "resid"}}, "save_frames": optional bool}}. At most 256 episodes per pass ("both": at most 128 per model, run on the same episodes and seeds, returning the paired difference with its standard error). Returns success (overall, by perturbation type and suite), how many task-perturbation failures did the original task instead, re-grasp-loop counts; per-episode records in WORK/roll_<k>.jsonl (instruction, success, orig_goal_reached, steps, grip_flips, eef_path_m, obj_moved_m, path of the executed actions .npy); with "record", per-episode activation summaries (mean over the episode's policy queries; VLM outputs averaged separately over base-image, wrist-image and instruction tokens, expert outputs over the action tokens) as .npy in WORK/roll_<k>_acts/ with index.json; with "save_frames", the first observation and every 50th step as .png in WORK/roll_<k>_frames/ and the exact policy inputs needed for probes.
 (b) PROBE (cheap, no simulation): {{"type": "probe", "model": "original" | "edited" | "both", "inputs": "<path of saved policy inputs from a rollout with save_frames, or a list of them>", "return": {{"layers": [expert layers], "vlm_layers": [...], "vision_layers": [...], "what": "attn_out" | "mlp_out" | "resid" | "attn_maps"}}}}: runs the policy on stored observations (up to 512 inputs) and returns the predicted action chunk and requested activations (per call) as .npy in WORK/probe_<k>/ (attention maps of the action expert over the prefix: at most 4 inputs, at most 4 layers).
Same episode and seed give the same result (deterministic). A rollout pass of 256 episodes takes on the order of 10 minutes.

"""

WORK = """HOW TO WORK.
1. Diagnose first: compare failing perturbed episodes with the same task's unperturbed or successful ones; look at the saved frames and action traces (which task does it actually perform, where does it reach), and at which internal features or attention patterns differ.
2. Localize causally: correlation between a component and success is not enough; show that changing it changes the behaviour as predicted, with a dose-response.
3. Measure with power: success over n episodes has a standard error of about 50/sqrt(n) points. Compare original and edited on the SAME episodes and seeds ("model": "both"), and keep an edit only if its gain is clearly larger than its standard error on episodes not used to choose it. Check for damage on perturbation types it already solves (object, semantic) before you finish.
4. Plan your passes; keep WORK/notes.md up to date (findings with numbers, edits tried and their measured effect, current best edit; it is shown to you again if the conversation is restarted); keep copies of good edits (e.g. copy WORK/edited to WORK/best_1/).
5. Print compact summaries, never whole arrays; outputs are ASCII only.

"""

_have = _sec("WHAT YOU HAVE.", "2. A forward-pass tool")
_have = _have.replace("BENCH/practice_variants.json lists the PRACTICE variants (suite, id, name, category, difficulty) - the only ones you may run; the held-out ones are not listed.",
                      "BENCH/practice_episodes.json lists the PRACTICE episodes (id, key, suite, ptype, task, init_idx, instruction, instruction_changed) - the only ones you may run; the held-out ones are not listed.")
_have = _have.replace("BENCH/paligemma_tokenizer.model", "BENCH/paligemma_tokenizer.model")
_rules = _sec("RULES.", "HOW THE SESSION RUNS.").replace("BENCH/practice_variants.json", "BENCH/practice_episodes.json") \
    .replace("in particular not the held-out variant list", "in particular not the held-out episode list")
TASK = (INTRO + _sec("THE MODEL", "THE BENCHMARK.") + BENCHMARK + _have + FWD + WORK + _rules
        + _b.TASK[_b.TASK.index("HOW THE SESSION RUNS."):])
PRELUDE = _b.PRELUDE


def setup(R):
    global PRELUDE
    assert os.path.exists(BENCH + "/practice_episodes.json")
    PRELUDE = _b.PRELUDE.replace("__CKPT__", CKPT).replace("__WORK__", R + "/work").replace("__BENCH__", BENCH)


def forward(req, R, k):
    spec = importlib.util.spec_from_file_location("fwd", os.path.join(H, "pro_forward.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.forward(req, R, k, target=TARGET)
