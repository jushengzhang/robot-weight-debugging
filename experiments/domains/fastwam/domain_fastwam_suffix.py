"""Weight-space programming domain: FastWAM under the LIBERO-Plus reference instruction format (task.language with the
variant suffix, e.g. "put the bowl on the plate view 69 0 100 0 0 initstate 0"), which drops it from ~70% to ~49%.
Same model, weights, kernel and forward tool as domain_fastwam; rollouts default to language="raw".
"""
import importlib.util, os

os.environ["WSP_LANGUAGE"] = "raw"  # read by fastwam_forward.py
_spec = importlib.util.spec_from_file_location("domain_fastwam_base", os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_fastwam.py"))
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)
KERNEL_PYTHON, PRELUDE, forward = _b.KERNEL_PYTHON, _b.PRELUDE, _b.forward


PRIOR = os.environ.get("WSP_PRIOR")  # run dir of an earlier session on this task: its notes and last candidate are given


def setup(R):
    global PRELUDE
    _b.setup(R)
    PRELUDE = _b.PRELUDE  # paths filled in by the base setup
    if PRIOR:
        import shutil
        dst = R + "/work/prior_candidate"
        if not os.path.isdir(dst):
            shutil.copytree(PRIOR + "/work/edited", dst)
            shutil.copy(PRIOR + "/work/notes.md", R + "/work/prior_notes.md")
TARGET = "instruction-suffix distraction"


def _section(name, nxt):
    t = _b.TASK
    return t[t.index(name):t.index(nxt)]


INTRO = """TASK. You are given the weights of FastWAM, a world-action model that controls a Franka robot arm in the LIBERO simulator (a released checkpoint, fine-tuned on the LIBERO demonstrations). The LIBERO-Plus benchmark gives the policy its task instruction with a machine-generated variant tag appended, exactly as its reference evaluation scripts do: e.g. "put the bowl on the plate view 69 0 100 0 0 initstate 0", "open the top drawer of the cabinet light 39", "pick up the ketchup and place it in the basket noise 10", "... tb 3", "... add 2", "... level2". The tag describes how the scene was perturbed; it carries no information about what the robot should do. With the clean instruction FastWAM succeeds on about 70% of episodes; with the tagged instruction it drops to about 49% (on a 210-episode sample: libero_goal 74% -> 26%, Camera 47% -> 17%, Robot Initial States 77% -> 33%, Sensor Noise 60% -> 33%). Your job is to make FastWAM IGNORE the variant tag, i.e. act on the tagged instruction as it would on the clean one, by EDITING ITS WEIGHTS DIRECTLY: find out how the tag reaches and corrupts the action computation, identify the components responsible, and repair them with targeted changes, without hurting performance with clean instructions or on the Language Instructions category (whose variants are paraphrased instructions without a tag, which the model must still follow).

"""

BENCH = """THE BENCHMARK. LIBERO-Plus: the 4 LIBERO suites (spatial, object, goal, long/libero_10) with 7 perturbation categories (Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise, Objects Layout) and difficulty levels 1-5. An episode succeeds if the task's goal predicate holds within the step limit (400 steps; 700 for libero_10). Every non-Language variant has a tag; the tag text depends on the category (camera: "view <5 numbers> initstate <n>", light: "light <n>", noise: "noise <n>", background: "tb <n>", layout: "add <n>" / "level<n>", robot initial state: "initstate <n>", ...). The instruction text is encoded by a frozen umt5-xxl (not editable): its token embeddings, including those of the tag tokens, are what each expert's cross-attention reads.

HOW IT WILL BE MEASURED. After you finish, the edited and the unedited model are evaluated on HELD-OUT LIBERO-Plus variants you never see (a fixed half of all variants, chosen by a hash), with fixed seeds: (1) success with the TAGGED instructions over all 7 categories (the goal: raise it toward the clean-instruction level), and (2) success with CLEAN instructions and on the Language Instructions category (must not drop; a drop of more than ~2 points counts against you). Only weight values change: same tensor names and shapes, same inference code, settings and instruction strings.

"""

TOOL_NOTE = """ LANGUAGE OPTION: every rollout takes an optional "language": "raw" (default: the tagged instruction, the condition being measured) or "strip" (the clean instruction, tag removed). Running the same variants and seeds with "raw" and "strip" gives you a matched clean/corrupted contrast for diagnosis; probes replay stored inputs exactly as recorded, including their instruction.
"""

WORK = """HOW TO WORK.
1. Diagnose first: compare rollouts of the same variants and seeds with "raw" and "strip" instructions; find where the two runs diverge (actions, failure modes), and which internal features differ between them (which layers' cross-attention outputs, which heads attend to the tag tokens, how much of the difference passes through the text cross-attention of each expert versus elsewhere).
2. Localize causally: correlation between a component and success is not enough; show that changing it changes the behaviour as predicted, with a dose-response.
3. Measure with power: success over n episodes has a standard error of about 50/sqrt(n) points. Compare original and edited on the SAME variants and seeds ("model": "both"), and keep an edit only if its gain is clearly larger than its standard error on variants not used to choose it. Check clean-instruction ("strip") performance and the Language Instructions category for damage before you finish.
4. Plan your passes; keep WORK/notes.md up to date (findings with numbers, edits tried and their measured effect, current best edit; it is shown to you again if the conversation is restarted); keep copies of good edits (e.g. copy WORK/edited to WORK/best_1/).
5. Print compact summaries, never whole arrays; outputs are ASCII only.

"""

_have = _section("WHAT YOU HAVE.", "HOW TO WORK.")
_have = _have.replace("The forward tool may not be online", TOOL_NOTE.strip() + "\nThe forward tool may not be online")
TASK = (INTRO + _section("THE MODEL.", "THE BENCHMARK.") + BENCH + _have + WORK
        + _section("RULES.", "HOW THE SESSION RUNS.") + _b.TASK[_b.TASK.index("HOW THE SESSION RUNS."):])

if PRIOR:
    TASK += """

PRIOR WORK. An earlier session (40 forward passes) worked on this same task. Its notes are in WORK/prior_notes.md (read them first; its pass numbers and WORK paths refer to that session, whose files are not available apart from these notes). Its last candidate edit, a gate on the action expert's text cross-attention queries/keys in many blocks (308 tensors), is in WORK/prior_candidate/<tensor name>.npy; it is NOT applied (WORK/edited starts empty). On that session's own validation it gave about +1.8 points on tagged instructions (paired SE about 1.4) with a small loss on paraphrased instructions, i.e. it closed only a small part of the ~20-point gap between tagged and clean instructions. Build on what it learned, but aim for a much larger, validated effect: find what the earlier attempt missed (e.g. other routes by which the tag reaches the action computation, the video expert's text cross-attention, or how the tag changes the umt5 embeddings of the task tokens themselves)."""
