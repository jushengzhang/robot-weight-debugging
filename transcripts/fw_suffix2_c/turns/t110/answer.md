The key-only edit barely changed arm motion, while value correction did help. I’ll now test the earlier one-tensor video text-output edit in simulation, without the action-expert addition that was present in the previous development rollout. This checks whether that addition obscured a useful video-only effect.
```python
clear_edits()
cand=W/"candidate_video_weighted_full"
for p in cand.glob("*.npy"):
    save_edit(p.name[:-4],torch.from_numpy(np.load(p)))
print("active edits",list(edited()))
req37=dict(req25)
(W/"request37.json").write_text(json.dumps(req37))
print(json.dumps(req37,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nKey-only35/36 comparison: value-only gap.7752, arm10.9054; key-only gap.9571, arm10.9987. Stronger evidence that video text values carry the useful correction. Restored previously saved one-tensor weighted_video_full (probe22) for development rollout37, without action edit. Request is same development96 + diagnostic language32 as25. Reserved validation sets remain untouched.\n")
```
