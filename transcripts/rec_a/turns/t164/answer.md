The final panel improved, but the full reserved validation set did not: 166/288 successes for the original versus 164/288 for the edit, with 15 rescues and 17 losses. This fails the precommitted acceptance rule. I will remove the edit, verify that the submitted weights are unchanged, and save the aggregate results.
```python
from pathlib import Path
from collections import defaultdict
import json
import math
import re
import numpy as np
import torch

w = Path(WORK)
panels = json.loads((w / "independent_panels.json").read_text())
all_pairs = []
pass_pairs = {}
excluded = {}

for k, requested in zip((48, 49, 50), panels[1:]):
    rows = [
        json.loads(line)
        for line in (w / f"roll_{k}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    indexed = {"original": {}, "edited": {}}
    for row in rows:
        if row.get("error") or not isinstance(row.get("success"), bool):
            continue
        key = (row["id"], row["grip"], row["seed"])
        model = row["model"]
        assert model in indexed
        assert key not in indexed[model], (k, model, key)
        indexed[model][key] = row

    requested_keys = {(r["id"], r["grip"], r["seed"]) for r in requested}
    matched = set(indexed["original"]) & set(indexed["edited"])
    assert matched <= requested_keys
    excluded[str(k)] = [list(key) for key in sorted(requested_keys - matched)]
    pairs = [(indexed["original"][key], indexed["edited"][key])
             for key in sorted(matched)]
    for original, candidate in pairs:
        assert original["level"] == candidate["level"]
        assert original["suite"] == candidate["suite"]
        assert original["success_at_t0"] == candidate["success_at_t0"]
    pass_pairs[k] = pairs
    all_pairs.extend(pairs)

assert [len(pass_pairs[k]) for k in (48, 49, 50)] == [102, 94, 92]
assert len({(o["id"], o["grip"], o["seed"]) for o, e in all_pairs}) == len(all_pairs)

def paired_summary(pairs):
    n = len(pairs)
    differences = np.array(
        [int(e["success"]) - int(o["success"]) for o, e in pairs],
        dtype=float,
    )
    mean_difference = float(differences.mean())
    paired_se = float(differences.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    scene_indices = defaultdict(list)
    for i, (o, e) in enumerate(pairs):
        scene_indices[o["id"]].append(i)
    cluster_residual_sums = np.array([
        float(differences[ix].sum() - len(ix) * mean_difference)
        for ix in scene_indices.values()
    ])
    g = len(scene_indices)
    cluster_se = (
        float(np.sqrt(g / (g - 1) * np.square(cluster_residual_sums).sum()) / n)
        if g > 1 else 0.0
    )
    return {
        "n": n,
        "scenes": g,
        "original_successes": sum(int(o["success"]) for o, e in pairs),
        "edited_successes": sum(int(e["success"]) for o, e in pairs),
        "gain_pp": 100 * mean_difference,
        "paired_se_pp": 100 * paired_se,
        "scene_cluster_se_pp": 100 * cluster_se,
        "rescues": int((differences == 1).sum()),
        "losses": int((differences == -1).sum()),
        "original_loop_failures": sum(
            int(not o["success"] and o["grip_flips"] >= 6) for o, e in pairs
        ),
        "edited_loop_failures": sum(
            int(not e["success"] and e["grip_flips"] >= 6) for o, e in pairs
        ),
        "already_solved_at_start": sum(int(o["success_at_t0"]) for o, e in pairs),
    }

overall = paired_summary(all_pairs)
active = paired_summary([
    (o, e) for o, e in all_pairs if not o["success_at_t0"]
])
by_level = {
    level: paired_summary([(o, e) for o, e in all_pairs if o["level"] == level])
    for level in ("L1", "L2", "L3", "L4")
}
by_suite = {
    suite: paired_summary([(o, e) for o, e in all_pairs if o["suite"] == suite])
    for suite in sorted({o["suite"] for o, e in all_pairs})
}

criteria = {
    "gain_exceeds_two_SE": overall["gain_pp"] > 2 * max(
        overall["paired_se_pp"], overall["scene_cluster_se_pp"]
    ),
    "fewer_loop_failures": (
        overall["edited_loop_failures"] < overall["original_loop_failures"]
    ),
    "losses_at_most_two_percent": overall["losses"] <= 0.02 * overall["n"],
}
assert not all(criteria.values())

# Preserve the rejected candidate only as an audit artifact, outside submitted edits.
removed_names = list(edited())
assert removed_names == [
    "paligemma_with_expert.gemma_expert.model.layers.17.self_attn.o_proj.weight"
]
torch.save(
    {name: load(name) for name in removed_names},
    w / "rejected_final_beta_minus2.pt",
)
for name in removed_names:
    drop_edit(name)
assert list(edited()) == []
for name in removed_names:
    assert torch.equal(load(name), load_original(name))

report = {
    "decision": "reject candidate; submit original weights with no edits",
    "candidate": "E17H1_visual_grip_retention_beta_minus2",
    "passes_used": 50,
    "requested_validation_pairs": sum(len(panel) for panel in panels[1:]),
    "excluded_episode_pairs": excluded,
    "overall": overall,
    "excluding_already_solved": active,
    "by_pass": {str(k): paired_summary(v) for k, v in pass_pairs.items()},
    "by_level": by_level,
    "by_suite": by_suite,
    "acceptance_criteria": criteria,
    "submitted_edits": list(edited()),
    "ordinary_task_status": (
        "No original-task evaluation was exposed. Submitted weights are unchanged, "
        "so no weight-induced performance change is claimed."
    ),
}
(w / "final_decision.json").write_text(json.dumps(report, indent=2))
(w / "current_candidate.json").write_text(json.dumps({
    "name": "none",
    "status": "final",
    "reason": "Frozen candidate failed the precommitted independent validation rule.",
}))
with (w / "candidate_log.jsonl").open("a") as f:
    f.write(json.dumps({
        "pass": 50,
        "candidate": report["candidate"],
        "decision": "rejected_and_reverted",
        "validation": overall,
    }) + "\n")

notes = (w / "notes.md").read_text()
notes = re.sub(
    r"## Pass budget\n.*?(?=\n## )",
    "## Pass budget\n- Forward passes used: 50/50. Final submitted edits: none.\n",
    notes, count=1, flags=re.S,
)
notes = re.sub(
    r"## Current edits\n.*?(?=\n## )",
    "## Current edits\n- None. All candidates reverted; original checkpoint submitted.\n",
    notes, count=1, flags=re.S,
)
notes += (
    "\n\n## Pass 50 and final decision\n"
    "- Frozen candidate final panel: 51/92 -> 55/92, 6 rescues / 2 losses; "
    "loop failures 31 -> 27.\n"
    f"- Reserved validation aggregate: {overall['original_successes']}/{overall['n']} "
    f"-> {overall['edited_successes']}/{overall['n']}; "
    f"gain {overall['gain_pp']:.3f} pp, paired SE {overall['paired_se_pp']:.3f} pp, "
    f"scene-cluster SE {overall['scene_cluster_se_pp']:.3f} pp.\n"
    f"- Rescues/losses: {overall['rescues']}/{overall['losses']}; "
    f"loop failures {overall['original_loop_failures']} -> "
    f"{overall['edited_loop_failures']}.\n"
    "- Two requested episode pairs were unavailable due to scene errors; "
    "288 matched pairs were analyzed. Already-solved starts are separately "
    "reported in final_decision.json.\n"
    "- Failed all precommitted acceptance criteria. Removed the final edit "
    "using drop_edit; verified edited() is empty and restored tensor equals original.\n"
    "- No successful repair established. Attention and action changes were causal, "
    "but none justified a claim of improved recovery. Original-task performance "
    "was not directly measured; unchanged submission preserves the original policy.\n"
)
(w / "notes.md").write_text(notes)

print("AGGREGATE VALIDATION:")
print(json.dumps(overall, indent=2))
print("EXCLUDING ALREADY-SOLVED STARTS:")
print(json.dumps(active, indent=2))
print("ACCEPTANCE:", json.dumps(criteria))
print("FINAL SUBMITTED EDITS:", list(edited()))
print("RESTORED TENSOR EXACTLY MATCHES ORIGINAL: True")
```
