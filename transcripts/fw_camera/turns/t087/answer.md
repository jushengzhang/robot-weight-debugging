Across all 474 independent camera episodes, the edit gains only 4 successes—too small to distinguish from noise. I’ll archive it as unconfirmed and restore the original readout. Next I’ll revisit the upstream diagnostics: a late readout correction may be too weak if viewpoint errors enter through attention earlier in the action stack.
```python
validation = paired_summary(paired_rows([20, 21, 22, 23]))
print("FINAL independent readout validation:", validation)
ledger = json.loads((root / "pass_ledger.json").read_text())
ledger.append({"pass": 23, "type": "rollout", "edit": "readout_camera_rank2_half",
               "purpose": "Independent camera validation D", **paired_summary(paired_rows([23]))})
(root / "pass_ledger.json").write_text(json.dumps(ledger, indent=2))
(root / "readout_validation_result.json").write_text(json.dumps(validation, indent=2))
drop_edit(wname)
print("Active edits after revert:", edited())

for fn in ["residual_diagnosis.npz", "camera_firstcall_pca.npz",
           "readout_protected_camera_subspace_broad.npz", "probe3_head_map_summary.npz"]:
    z = np.load(root / fn)
    print(fn, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})

print("Action block 11 tensor names/shapes:")
for n in NAMES:
    if ("action" in n and (".blocks.11." in n or ".head." in n or ".norm" in n and ".blocks." not in n)):
        print(n, SHAPES[n])
print("Candidate directories:", [p.name for p in (root / "candidates").iterdir()])
for pname in ("probe_2", "probe_3", "probe_10", "probe_12"):
    p = root / pname
    print(pname, [x.name for x in list(p.iterdir())[:5]] if p.exists() else "missing")
print("Partition structure:", str(json.loads((root / "partition.json").read_text()))[:700])
with (root / "notes.md").open("a") as f:
    f.write("\n## Pass23: readout candidate not confirmed\n"
            f"- D:34/90 vs39/90 (+5.556pp); all independent panels: {validation}.\n"
            "- +0.844pp on 474 independent episodes, below SE; do NOT promote discovery gain.\n"
            "- Archived candidate unchanged; reverted to ORIGINAL (no active edits).\n"
            "- Revisit upstream attention mechanism using discovery activations only; budget17 passes left.\n")
```
