The final candidate gained only 6 successes across 640 paired episodes: +0.94 percentage points, equal to its paired standard error. It fails the prespecified acceptance rule. I’ll remove the projector edit and save the final audit rather than submit an unvalidated repair.
```python
from collections import defaultdict

all_pairs = [p for k in range(36, 41) for p in read_pairs(k)]
overall = paired_summary(all_pairs)
assert overall["n"] == 640
assert overall["original"] == 373 and overall["edited"] == 379

# The frozen acceptance rule already fails its paired-SE requirement.
assert overall["delta_pp"] <= 2 * overall["paired_se_pp"]
removed_names = list(edited())
for name in removed_names:
    drop_edit(name)
assert list(edited()) == []

by_ptype = {
    ptype: paired_summary([(a, b) for a, b in all_pairs if a["ptype"] == ptype])
    for ptype in sorted({a["ptype"] for a, b in all_pairs})
}
assert all(s["n"] == 128 for s in by_ptype.values())

practice = json.loads((Path(BENCH) / "practice_episodes.json").read_text())
if isinstance(practice, dict):
    if "episodes" in practice:
        practice = practice["episodes"]
    elif "practice_episodes" in practice:
        practice = practice["practice_episodes"]
    else:
        practice = list(practice.values())
metadata = {int(r["id"]): r for r in practice}

delta = np.array(
    [int(b["success"]) - int(a["success"]) for a, b in all_pairs], dtype=float
)
cluster_sums = defaultdict(float)
for (a, b), residual in zip(all_pairs, delta - delta.mean()):
    m = metadata[int(a["id"])]
    cluster_sums[(m["suite"], m["task"], m["init_idx"])] += float(residual)
g = len(cluster_sums)
cluster_se_pp = float(
    100 * np.sqrt(
        (g / (g - 1)) * sum(v * v for v in cluster_sums.values()) / len(delta) ** 2
    )
)

criteria = {
    "overall_gt_two_SE": bool(
        overall["delta_pp"] > 2 * max(overall["paired_se_pp"], cluster_se_pp)
    ),
    "object_plus_semantic_nonnegative": bool(
        by_ptype["object"]["delta_pp"] + by_ptype["semantic"]["delta_pp"] >= 0
    ),
    "object_drop_at_most_2pp": bool(by_ptype["object"]["delta_pp"] >= -2),
    "semantic_drop_at_most_2pp": bool(by_ptype["semantic"]["delta_pp"] >= -2),
    "environment_nonnegative": bool(by_ptype["environment"]["delta_pp"] >= 0),
}
assert not all(criteria.values())

audit = {
    "final_selection": "original_checkpoint_no_active_edits",
    "rejected_candidate": plan["candidate"],
    "confirmation_passes": list(range(36, 41)),
    "seed": plan["seed"],
    "overall": overall,
    "by_ptype": by_ptype,
    "cluster_grouping": ["suite", "task", "init_idx"],
    "cluster_count": g,
    "cluster_se_pp": cluster_se_pp,
    "acceptance_checks": criteria,
    "removed_tensors": removed_names,
    "active_edits": list(edited()),
    "original_task_validation": "No unperturbed practice episodes were available.",
}
(work / "final_decision.json").write_text(json.dumps(audit, indent=2) + "\n")

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Final confirmation batch5 (pass40) and submission\n"
        "Batch5:67/128->70/128,4wins/1loss. "
        "All640:373->379 successes,21wins/15losses; "
        f"delta={overall['delta_pp']:.4f}pp, "
        f"pairedSE={overall['paired_se_pp']:.4f}pp, "
        f"groupedClusterSE={cluster_se_pp:.4f}pp (G={g}).\n"
        "Frozen acceptance rule FAILED. Removed both projector edits using drop_edit; "
        "verified edited() is empty. FINAL SUBMISSION: ORIGINAL CHECKPOINT, NO EDITS. "
        "No validated robustness gain was found. Full audit:final_decision.json.\n"
    )

print("Overall:", overall)
print("Grouped cluster SE (pp):", round(cluster_se_pp, 4), "clusters:", g)
for ptype, summary in by_ptype.items():
    print(
        f"{ptype:12s} {summary['original']:3d}/{summary['n']} -> "
        f"{summary['edited']:3d}/{summary['n']}; "
        f"delta={summary['delta_pp']:+.3f}pp, "
        f"pairedSE={summary['paired_se_pp']:.3f}pp"
    )
print("Acceptance checks:", criteria)
print("FINAL ACTIVE EDITS:", list(edited()))
print("Audit saved:", str(work / "final_decision.json"))
```
