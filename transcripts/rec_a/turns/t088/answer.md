Failed loops have more disagreement between the next five predicted gripper commands and the commands actually issued after replanning (18.0% versus 10.8% in successes). This is not proof of a cause, but it motivates checking the late action-token attention heads that amplify gripper decisions, rather than further weakening visual feedback.
```python
for k in [2,3,5,8,10]:
    print("probe", k, [p.name for p in (W/f"probe_{k}").iterdir()][:12])
gates=np.load(W/"inferred_residual_multipliers.npz")
print("gate keys", {k:gates[k].shape for k in gates.files})
print("action weights", [(n,SHAPES[n]) for n in NAMES if n.startswith("action_") or n.endswith("norm.dense.weight")][-10:])
print("grip MLP artifact", brief(torch.load(W/"gripper_mlp_channels.pt",weights_only=False,map_location="cpu")))
for k in [4,15,20,23]:
    js=json.loads((W/f"probe_{k}"/"results.json").read_text())
    print(k,"n",len(js),"first",js[0])
```
