The higher-rank video edit is worse and causes a large clean-gripper deviation on one camera example, so I’ll discard it. I’ll restore the better two-tensor candidate and run paired development rollouts, including the 32 diagnostic paraphrases as an early safety check.
```python
clear_edits()
for p in (W/"candidate_weighted_video1_actionhalf").glob("*.npy"):
    save_edit(p.name[:-4],torch.from_numpy(np.load(p)))
devmix=splits["development"]+splits["language_diag"]
req25={"type":"rollout","model":"both","language":"raw",
       "episodes":[{k:e[k] for k in ["suite","id","seed"]} for e in devmix]}
(W/"request25.json").write_text(json.dumps(req25))
print(json.dumps(req25,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nPass24 rank256 video rejected: gap ratio.8686, clean drift.01948 (camera gripper outlier); worse than rank128. Restored candidate_weighted_video1_actionhalf (2 tensors) for paired development96 + diagnostic language32 rollout25. Independent validation splits remain unused.\n")
```
