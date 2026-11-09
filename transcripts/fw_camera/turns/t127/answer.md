The full camera validation produced only 8 additional successes in 1,242 paired episodes (+0.64 points), smaller than its standard error. This fails the preset retention criterion. Before the final report, I need to remove the candidate edit, verify restoration of the original weights, and save the decision record.
```python
final_camera_passes = [20, 21, 22, 23, 28, 29, 30, 31, 39, 40]
final_camera_rows = read_pairs(final_camera_passes)
assert len(final_camera_rows) == 1242
assert all(r["category"] == "Camera Viewpoints" for r in final_camera_rows)
assert len({(r["suite"], r["id"], r["seed"])
            for r in final_camera_rows}) == 1242
assert Counter((r["suite"], r["id"], r["seed"])
               for r in final_camera_rows if r["run"] == 40) == Counter(
    (r["suite"], r["id"], r["seed"])
    for r in plan["camera_panels"]["40"]
)

final_camera = paired_summary(final_camera_rows)
assert final_camera["variants"] == 602
final_camera["original_pct"] = 100 * final_camera["original"] / final_camera["n"]
final_camera["edited_pct"] = 100 * final_camera["edited"] / final_camera["n"]
final_camera["ci95_cluster_pp"] = [
    final_camera["diff_pp"] - 1.96 * final_camera["cluster_se_pp"],
    final_camera["diff_pp"] + 1.96 * final_camera["cluster_se_pp"]
]

final_safety = {
    cat: paired_summary(read_pairs(runs))
    for cat, runs in safety_runs.items()
}
camera_ok = final_camera["diff_pp"] > 1.96 * final_camera["cluster_se_pp"]
safety_ok = all(s["diff_pp"] >= -2.0 for s in final_safety.values())
retain = bool(camera_ok and safety_ok)

name = "mot.mixtures.action.head.weight"
assert list(edited()) == [name]
candidate_tensor = load(name).clone()
candidate_hash = hashlib.sha256(
    candidate_tensor.contiguous().numpy().tobytes()
).hexdigest()
assert candidate_hash == lock["tensor_sha256"]

# Preserve the tested candidate outside the submitted edits directory.
archive_path = W / "final_tested_candidate_archive.pt"
torch.save({name: candidate_tensor}, archive_path)

if not retain:
    for tensor_name in list(edited()):
        drop_edit(tensor_name)
    assert list(edited()) == []
    assert torch.equal(load(name), load_original(name))
    choice = "original"
else:
    choice = lock["candidate"]

report = {
    "forward_passes_used": 40,
    "tested_candidate": lock["candidate"],
    "candidate_sha256": candidate_hash,
    "candidate_archive": archive_path.name,
    "camera_passes": final_camera_passes,
    "camera": final_camera,
    "safety": final_safety,
    "noncamera_pooled": paired_summary(read_pairs([32, 33, 34, 35, 36, 37, 38])),
    "camera_rule_met": bool(camera_ok),
    "safety_point_estimate_rule_met": bool(safety_ok),
    "safety_noninferiority_proven": False,
    "final_choice": choice,
    "submitted_edits": list(edited()),
    "reason": (
        "Retained under fixed criteria." if retain else
        "Independent camera gain did not exceed 1.96 times variant-cluster SE; "
        "restored original weights under the predeclared fallback."
    )
}
(W / "final_validation_report.json").write_text(json.dumps(report, indent=2))
(W / "final_choice.json").write_text(json.dumps({
    "choice": choice,
    "submitted_edits": list(edited()),
    "report": "final_validation_report.json"
}, indent=2))

with (W / "notes.md").open("a") as f:
    f.write("\n## FINAL: all40 forward passes complete\n")
    f.write("- Camera validation: " + json.dumps(rounded(final_camera)) + "\n")
    f.write("- Camera rule met: " + str(camera_ok) +
            "; noncamera point-estimate rule met: " + str(safety_ok) + ".\n")
    f.write("- Final choice: " + choice + ". Active edits: " +
            json.dumps(list(edited())) + ".\n")
    f.write("- Candidate archived outside submitted edits; "
            "no confirmed camera-robustness improvement. "
            "Complete results in final_validation_report.json.\n")

print("Final camera validation:", rounded(final_camera))
print("Noncamera pooled:", rounded(report["noncamera_pooled"]))
print("Camera criterion met:", camera_ok)
print("Safety point-estimate criterion met:", safety_ok)
print("FINAL CHOICE:", choice)
print("SUBMITTED EDITS:", list(edited()))
print("Original head restored exactly:",
      torch.equal(load(name), load_original(name)))
```
