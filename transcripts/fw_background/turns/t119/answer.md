I’ll verify that the intended MLP components were actually reduced, then measure the downstream action effect separately for failed backgrounds, successful backgrounds, and light controls. This will distinguish a successful weight intervention from a useful policy repair.
```python
p29 = probe_results(29)
A29 = np.stack([np.load(r["action"]) for r in p29])
acts29 = [dict(np.load(r["activations"])) for r in p29]
summary29 = {
    "pass":29,"dose":.5,
    "background_action_change_rms":rms(A29[:54]-A26[:54]),
    "light_action_change_rms":rms(A29[54:]-A26[54:]),
    "successful_bg_action_change_rms":rms(A29[success_bg_indices]-A26[success_bg_indices]),
    "failed_bg_action_change_rms":rms(A29[fail_idx]-A26[fail_idx]),
    "bg_light_discrepancy_ratio":rms(matched_differences(A29))/rms(matched_differences(A26)),
    "failure_control_discrepancy_ratio":
        rms(A29[F[:,0]]-A29[F[:,1]])/rms(A26[F[:,0]]-A26[F[:,1]])
}
for L in [1,2]:
    key = f"mean::action.{L}.mlp_out::action"
    orig = np.stack([r[key] for r in mlp_acts])
    edit = np.stack([r[key] for r in acts29])
    direction = mlp_causal[f"target{L}"]
    muL = mlp_causal[f"mean{L}"]
    summary29[f"layer{L}_target_projection_ratio"] = \
        rms((edit-muL)@direction)/rms((orig-muL)@direction)
key29 = "mean::action.29.mlp_out::action"
summary29["late_mlp_change_rms"] = rms(
    np.stack([r[key29] for r in acts29])-np.stack([r[key29] for r in mlp_acts]))
print(json.dumps(summary29,indent=2))
(W/"probe29_summary.json").write_text(json.dumps(summary29,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass29: early MLP fixed-input causal test\n")
    f.write("- "+json.dumps(summary29)+"\n")
```
