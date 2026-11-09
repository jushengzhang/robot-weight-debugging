The rollout residual summaries are identical across the requested layers, so I will not use their magnitudes to localize a fault. I also found no successful object/semantic match for one chosen scene. I will use the tomato-sauce and salad-dressing scenes instead and inspect token-level attention.
```python
target = [next(r for r in r1 if r["id"]==i) for i in [1946,1923,1562,1490]]
diag_inputs = [first_input(r) for r in target]
print("Probe cases:", [(r["id"],r["ptype"],r["success"]) for r in target])
print(json.dumps(diag_inputs))
keys_v = [f"mean::vlm.{l}.resid::base" for l in [0,3,6,9,12,15,17]]
print("Summary equality checks:",
      {l:bool(np.array_equal(a0[keys_v[0]],a0[f"mean::vlm.{l}.resid::base"])) for l in [3,6,9,12,15,17]})
print("Example summary stats:",
      {k:{"min":float(a0[k].min()),"max":float(a0[k].max()),"std":float(a0[k].std()),
          "nonzero":int(np.count_nonzero(a0[k]))} for k in [keys_v[0],"mean::expert.0.resid::action"]})
(W/"diag_inputs.json").write_text(json.dumps(diag_inputs))
with (W/"notes.md").open("a") as f:
    f.write("Rollout resid summaries are identical across requested layers; do not infer layer localization from these.\n"
            "Concrete old-goal examples: task1946 BBQ request moves tomato sauce 0.359m, BBQ 0; "
            "task1562 tomato request moves salad dressing 0.352m, tomato 0.\n")
```
