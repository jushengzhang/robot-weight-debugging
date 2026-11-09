I will compare the full-dose result with the half dose, then run the prepared orthogonal control. This will tell us whether any small action change is specific to the texture-related direction rather than a generic consequence of perturbing the weights.
```python
s32 = summarize_late_probe(32, 1.0)
a31, an31, _, _ = load_probe(31)
a32, an32, _, _ = load_probe(32)
d31 = (a31-orig_a).reshape(-1)
d32 = (a32-orig_a).reshape(-1)
s32["action_change_ratio_full_to_half"] = rms(d32)/max(rms(d31),1e-15)
s32["half_full_change_cosine"] = float(
    np.dot(d31,d32)/(np.linalg.norm(d31)*np.linalg.norm(d32)+1e-15))
s32["failure_half_to_full_increment_rms"] = rms((a32-a31)[failed_bg])
(W / "probe32_summary.json").write_text(json.dumps(s32, indent=2))
print(json.dumps(s32, indent=1))
print("Set norm-matched full-dose control:", set_late_mlp_edit(1.0, "control"))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass32: late-MLP full-dose probe\n- "+json.dumps(s32)+"\n")
```
