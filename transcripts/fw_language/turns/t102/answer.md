I will compare the replay results with the original probe, using the first denoising step to test localization. Later steps can change upstream layers through feedback from the altered action estimate, so only the first step is a clean “unchanged before layer 11” check.
```python
p22_results = json.loads((work / "probe_22" / "results.json").read_text())
assert len(p22_results) == len(p4_results)
o_actions, e_actions = [], []
old_proj, new_proj, first_old_proj, first_new_proj = [], [], [], []
p22_acts = []
layer10_first_rms, layer10_mean_rms = [], []
for old, new in zip(p4_results, p22_results):
    assert Path(old["input"]).name == Path(new["input"]).name
    assert work_path(old["input"]) == work_path(new["input"])
    o_actions.append(np.load(work_path(old["action"])))
    e_actions.append(np.load(work_path(new["action"])))
    with np.load(work_path(old["activations"])) as zo, np.load(work_path(new["activations"])) as ze:
        kmean = "mean::action.10.resid::action"
        kcall = "calls::action.10.resid::action"
        layer10_mean_rms.append(float(np.sqrt(np.mean((ze[kmean]-zo[kmean])**2))))
        layer10_first_rms.append(float(np.sqrt(np.mean((ze[kcall][0]-zo[kcall][0])**2))))
        old_proj.append(float((zo["mean::action.11.resid::action"] -
                               zo["mean::action.10.resid::action"]) @ text_u.numpy()))
        new_proj.append(float((ze["mean::action.11.resid::action"] -
                               ze["mean::action.10.resid::action"]) @ text_u.numpy()))
        first_old_proj.append(float((zo["calls::action.11.resid::action"][0] -
                                     zo["calls::action.10.resid::action"][0]) @ text_u.numpy()))
        first_new_proj.append(float((ze["calls::action.11.resid::action"][0] -
                                     ze["calls::action.10.resid::action"][0]) @ text_u.numpy()))
o_actions, e_actions = np.stack(o_actions), np.stack(e_actions)
def pair_difference(a):
    a = np.asarray(a)
    return a[1::2]-a[0::2]
def projection_contrast_ratio(new, old):
    dn, do = pair_difference(new), pair_difference(old)
    return float(np.dot(dn, do)/np.dot(do, do))
old_gaps = np.sqrt(np.mean(pair_difference(o_actions)**2, axis=(1,2)))
new_gaps = np.sqrt(np.mean(pair_difference(e_actions)**2, axis=(1,2)))
shifts = np.sqrt(np.mean((e_actions-o_actions)**2, axis=(1,2)))
probe22_summary = {
    "layer10_first_step_mean_rms_shift": float(np.mean(layer10_first_rms)),
    "layer10_all_steps_mean_rms_shift": float(np.mean(layer10_mean_rms)),
    "L11_mode_contrast_ratio_first_step": projection_contrast_ratio(first_new_proj, first_old_proj),
    "L11_mode_contrast_ratio_step_mean": projection_contrast_ratio(new_proj, old_proj),
    "mean_action_pair_gap_original": float(old_gaps.mean()),
    "mean_action_pair_gap_edited": float(new_gaps.mean()),
    "successful_prompt_action_shift_mean": float(shifts[::2].mean()),
    "failed_prompt_action_shift_mean": float(shifts[1::2].mean()),
    "pairs_with_smaller_action_gap": int((new_gaps < old_gaps).sum()),
    "pair_gap_original": old_gaps.tolist(),
    "pair_gap_edited": new_gaps.tolist()
}
print(json.dumps(probe22_summary, indent=2))
(work / "probe22_comparison.json").write_text(json.dumps(probe22_summary, indent=2))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 22: direct text-mode fixed-input causal probe\n"
            "- " + json.dumps(probe22_summary) + "\n"
            "- Activation attenuation alone is not proof of a useful repair. "
            "Proceed to the predeclared paired discovery rollout.\n")
```
