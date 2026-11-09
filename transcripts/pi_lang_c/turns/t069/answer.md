The tool only accepts unmodified rollout inputs, so it rejected those diagnostic copies without spending a pass. I will use the genuine literal-instruction rollouts instead. Original-versus-edited comparisons will still hold inputs fixed; comparisons to literal controls will be treated as matched-task, not pixel-identical, evidence.
```python
literal_records={r["instruction"]:r for r in read_roll(2)}
for c in probe_cases:
    if c["label"].startswith("literal_"):
        r=literal_records[c["instruction"]]
        c["path"]=str(pathlib.Path(r["frames_dir"])/"t0000_inputs.pt")
        assert pathlib.Path(c["path"]).exists()
shutil.rmtree(ctrl_dir)
json.dump(probe_cases,(W/"F_controlled_probe_cases.json").open("w"),indent=2)
print("ROLLOUT INPUT PATHS",json.dumps([c["path"] for c in probe_cases],separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nDiagnostic prompt-copy probe was refused (no pass spent): inputs must be unmodified rollout files. "
            "Copies removed. Use genuine literal-rollout inputs instead. Original-vs-edited comparisons "
            "are identical-input; paraphrase-vs-literal comparisons have matched task but may differ in images. "
            "Do not claim pixel-controlled counterfactual results.\n")
```
