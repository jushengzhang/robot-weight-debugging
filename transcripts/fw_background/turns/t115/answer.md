I’ll quantify the fixed-input effect against both the original residual probe and a repeated original action probe. This separates a genuine patch-edit effect from replay variability and checks whether texture sensitivity decreases without similarly disturbing the controls.
```python
p27 = probe_results(27)
A26 = np.stack([np.load(r["action"]) for r in p26])
A27 = np.stack([np.load(r["action"]) for r in p27])
assert [r["input"] for r in p26] == [r["input"] for r in p27]
base_diff = matched_differences(A26)
edit_diff = matched_differences(A27)
fail_idx = sorted(set(F[:,0].tolist()))
summary27 = {
    "pass":27,"dose":.25,
    "original_replay_rms":rms(A26-A0),
    "background_action_change_rms":rms(A27[:54]-A26[:54]),
    "light_action_change_rms":rms(A27[54:]-A26[54:]),
    "successful_bg_action_change_rms":rms(A27[success_bg_indices]-A26[success_bg_indices]),
    "failed_bg_action_change_rms":rms(A27[fail_idx]-A26[fail_idx]),
    "bg_light_discrepancy_ratio":rms(edit_diff)/rms(base_diff),
    "failure_control_discrepancy_ratio":
        rms(A27[F[:,0]]-A27[F[:,1]])/rms(A26[F[:,0]]-A26[F[:,1]]),
    "bg_light_discrepancy_rms_original":rms(base_diff),
    "bg_light_discrepancy_rms_edited":rms(edit_diff)
}
acts27 = [dict(np.load(r["activations"])) for r in p27]
resid_stats27 = {}
for L in [0,1,2,10,20,29]:
    key = f"mean::action.{L}.resid::action"
    B = np.stack([r[key] for r in acts19])
    E = np.stack([r[key] for r in acts27])
    bd,ed = matched_differences(B),matched_differences(E)
    direction = mean_shift_data[f"u{L}"]
    resid_stats27[L] = {
        "total_contrast_ratio":rms(ed)/rms(bd),
        "protected_texture_direction_ratio":rms(ed@direction)/rms(bd@direction),
        "light_residual_change_rms":rms(E[54:]-B[54:])
    }
summary27["residuals"] = resid_stats27
print(json.dumps(summary27,indent=2))
(W/"probe27_summary.json").write_text(json.dumps(summary27,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass27: patch DC fixed-input test\n")
    f.write("- "+json.dumps(summary27)+"\n")
```
