The current edit is a single rank-one addition to expert layer 15, head 2; it is not a global action or tensor scaling. I will verify its magnitude and load the screening episode IDs, then run the paired rollout test.
```python
name15 = "paligemma_with_expert.gemma_expert.model.layers.15.self_attn.o_proj.weight"
v = cue["value_direction"].float()
u = cue["write_direction"].float()
delta = load(name15) - load_original(name15)
expected = torch.zeros_like(delta)
expected[:, 512:768] = 2.0 * torch.outer(u, v)
print("Edit delta norm", delta.norm().item(), "construction error", (delta-expected).norm().item())
print("Original block norm", load_original(name15)[:,512:768].norm().item())
print("Screen episodes:", (W/"screen_episodes.json").read_text())
for k in [26,27]:
    pp = W/f"probe_{k}"
    print("probe", k, [p.name for p in pp.iterdir()][:8])
    for p in pp.glob("*.json"):
        j=json.loads(p.read_text())
        print(p.name, brief(j if not isinstance(j, list) else j[:1]))
manifest = json.loads((W/"physical_grip_probe_manifest.json").read_text())
print("physical manifest structure", brief(manifest))
```
