The beta=2 retreat edit failed: it rescued no episodes and lost five, so I will revert it. I will examine the affected trajectories and the original loops to distinguish “needs to retreat” from “already retreats but returns to the wrong place.”
```python
for n in list(edited()):
    drop_edit(n)
(W/"current_candidate.json").write_text(json.dumps({"name":"none"}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 28: empty-cue upward retreat rejected\n"
            "- Beta=2 paired screen: 31/64 -> 26/64, -7.8 pp (SE 3.4), 0 rescues / 5 losses.\n"
            "- Loop failures 21 -> 23; mean flips 12.62 -> 14.64. Reverted.\n"
            "- Visual selectivity and Z action causality did not imply a useful recovery behavior.\n"
            "- Forward budget now 28/50; current edits none.\n")
def read_roll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
r28 = read_roll(28)
print("Record example:", r28[0])
print("Catalog first:", brief(json.loads((W/"observation_catalog.json").read_text())[:1]))
print("Input example keys:")
inp0 = torch.load(manifest[0]["input"], weights_only=False, map_location="cpu")
print(brief(inp0))
print("Probe acts keys/shapes", {k:np.load(W/"probe_27/probe_orig_0_acts.npz")[k].shape for k in np.load(W/"probe_27/probe_orig_0_acts.npz").files})
```
