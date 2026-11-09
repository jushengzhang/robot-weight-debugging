The final tagged result is 506 to 516 successes out of 1,024: only +0.98 points, smaller than its paired standard error. The clean-instruction loss also exceeds the stated tolerance. I am rejecting the candidate and removing all edits so the submission does not retain that regression.
```python
import shutil

# Recompute all preplanned validation results without dropping any batches.
final_pairs = {}
final_passes = {}
for entry in validation_plan:
    pid = entry["pass"]
    pairs = read_pairs(pid)
    assert len(pairs) == 128
    requested = {
        (e["suite"], int(e["id"]), int(e["seed"]))
        for e in entry["request"]["episodes"]
    }
    assert {p[0] for p in pairs} == requested, ("request mismatch", pid)
    condition = "language" if entry["split"].startswith("language_") else entry["condition"]
    assert all(
        (p[1]["category"] == "Language Instructions") == (condition == "language")
        for p in pairs
    )
    final_pairs.setdefault(condition, []).extend(pairs)
    final_passes.setdefault(condition, []).append(pid)

final_stats = {}
for condition, pairs in final_pairs.items():
    assert len({p[0] for p in pairs}) == len(pairs)
    final_stats[condition] = summarize_pairs(pairs)
    final_stats[condition]["passes"] = final_passes[condition]

assert {k: v["n"] for k, v in final_stats.items()} == {
    "raw": 1024, "strip": 768, "language": 640
}
gain_supported = final_stats["raw"]["ci95_pp"][0] > 0
clean_safe = final_stats["strip"]["diff_pp"] >= -2.0
language_safe = final_stats["language"]["diff_pp"] >= -2.0
assert not (gain_supported and clean_safe and language_safe)

# Keep the rejected candidate for reproducibility, outside the submitted directory.
names_to_remove = list(edited())
assert set(names_to_remove) == set(frozen_manifest)
archive = root / "candidates" / "hybrid_r256_d1_t05_g128"
archive.mkdir(parents=True, exist_ok=True)
for name in names_to_remove:
    src = root / "edited" / (name + ".npy")
    dst = archive / (name + ".npy")
    if not dst.is_file() or sha256_file(dst) != frozen_manifest[name]:
        shutil.copy2(src, dst)
    assert sha256_file(dst) == frozen_manifest[name], name

# Final choice: original weights, with no residual instrumentation or repair edits.
for name in names_to_remove:
    drop_edit(name)
assert len(edited()) == 0
edit_dir = root / "edited"
assert not edit_dir.exists() or not list(edit_dir.iterdir())

decision = {
    "submission": "original_weights",
    "submitted_edit_count": 0,
    "rejected_candidate_tensor_count": len(names_to_remove),
    "rejected_candidate_archive": str(archive),
    "validation": final_stats,
    "criteria": {
        "tagged_gain_supported": bool(gain_supported),
        "clean_loss_within_2pp": bool(clean_safe),
        "language_loss_within_2pp": bool(language_safe)
    },
    "reason": "Tagged benefit is not established and clean-instruction loss exceeds tolerance."
}
(root / "final_decision.json").write_text(
    json.dumps(decision, indent=2, ensure_ascii=True) + "\n"
)
(root / "final_validation_summary.json").write_text(
    json.dumps(final_stats, indent=2, ensure_ascii=True) + "\n"
)

text = (root / "notes.md").read_text()
start = text.index("## Current state")
end = text.index("\n## Plan", start)
text = (
    text[:start]
    + "## Current state\n"
    + "FINAL: all50 forward passes completed. Rejected the frozen308-tensor candidate. "
      "ALL edits removed; edited() and WORK/edited verified empty. "
      "Submitted model is the original checkpoint. Full results: final_decision.json "
      "and final_validation_summary.json. Rejected snapshot remains only in "
      "candidates/hybrid_r256_d1_t05_g128, outside the submission directory.\n"
    + text[end:]
)
text += "\n## Final validation and decision\n" + json.dumps(decision, ensure_ascii=True) + "\n"
text += (
    "No validated repair was found. Diagnostic action-distance improvement did not "
    "translate into a reliable rollout benefit. No post-validation tuning or "
    "unvalidated alternative was submitted.\n"
)
(root / "notes.md").write_text(text)

print(json.dumps(final_stats, indent=1, ensure_ascii=True))
print("FINAL SUBMISSION: original weights")
print("Edited tensors:", len(edited()))
print("WORK/edited empty:", not edit_dir.exists() or not list(edit_dir.iterdir()))
```
