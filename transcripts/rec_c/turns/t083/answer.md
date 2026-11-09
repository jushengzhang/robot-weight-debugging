The independent test did not reproduce the gain: 12 wins and 12 losses, with more loop failures. I will remove that edit and examine two narrower possibilities in the saved data: coupling between gripper and arm commands, and common-mode key features that may divert attention away from the current scene.
```python
for name in list(edited()):
    drop_edit(name)
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass30: independent aperture-gain validation\n"
            "128 paired episodes:77 vs77,12 wins/12 losses (0.0+/-3.8pp); loops41 vs38. "
            "Development gain did not replicate. Removed aperture edit; no validated repair yet.\n")
def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
r1 = read_roll(1)
print("roll record example", str(r1[0])[:2300])
for d in ["fwd", "probe_3", "probe_8", "probe_15", "probe_19", "probe_20", "probe_26", "roll_1_frames"]:
    p = W / d
    print(d, [(q.name, q.stat().st_size if q.is_file() else "dir")
              for q in list(p.iterdir())[:8]] if p.exists() else "absent")
print("splits structure", str(json.loads((W/"splits.json").read_text()))[:1600])
ain = load_original("action_in_proj.weight")
aout = load_original("action_out_proj.weight")
ain_corr = torch.nn.functional.normalize(ain[:,:7], dim=0).T @ torch.nn.functional.normalize(ain[:,:7], dim=0)
aout_corr = torch.nn.functional.normalize(aout[:7], dim=1) @ torch.nn.functional.normalize(aout[:7], dim=1).T
print("action input col norms", ain[:,:7].norm(dim=0).tolist())
print("action output row norms", aout[:7].norm(dim=1).tolist())
print("input grip vs xyz/rot cos", np.round(ain_corr[6,:6].numpy(),3).tolist())
print("output grip vs xyz/rot cos", np.round(aout_corr[6,:6].numpy(),3).tolist())
print("output bias first7", load_original("action_out_proj.bias")[:7].tolist())
print("probe dirs", sorted(p.name for p in W.glob("probe_*")))
```
