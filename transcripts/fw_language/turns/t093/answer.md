The 1.5x source-mode edit produces 5 rescues and 5 regressions, with no net gain. I will measure its activation effect and test the predeclared 2.0x dose to distinguish a weak intervention from an unhelpful mechanism.
```python
p20 = paired_records([20])
s20 = paired_summary(p20)
assert {(k[0], k[1]) for k in p20} == discovery_keys
activation_changes = {}
for layer in [22, 23, 24]:
    vals = []
    for pair in p20.values():
        paths = [work_path(pair[m]["activations"]) for m in ("original", "edited")]
        with np.load(paths[0]) as ao, np.load(paths[1]) as ae:
            key = f"mean::video.{layer}.resid::agent"
            if key in ao and key in ae:
                vals.append([float(ao[key][1938]), float(ae[key][1938]),
                             float(np.sqrt(np.mean((ae[key]-ao[key])**2)))])
    v = np.asarray(vals)
    activation_changes[str(layer)] = {
        "n": len(v), "original_ch1938": float(v[:, 0].mean()),
        "edited_ch1938": float(v[:, 1].mean()),
        "mean_residual_shift_rms": float(v[:, 2].mean())
    }
print("Pass 20 paired result:", s20)
print("Episode-averaged activations (trajectory-confounded):", activation_changes)
(work / "source_gain15_results.json").write_text(
    json.dumps({"paired": s20, "activation_changes": activation_changes}, indent=2))
print("Next dose:", set_source_mode_gain(2.0))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 20: source-mode gain 1.5, no net improvement\n"
            "- Original discovery variants: " + json.dumps(s20) + "\n"
            "- Episode-mean channel-1938 changes: " +
            json.dumps(activation_changes) + "; trajectory-confounded, not fixed-input proof.\n"
            "- Apply the predeclared 2.0x leading-mode dose for the next causal test. "
            "No edits to other heads or modes.\n")
```
