The conditional query edit did not show the predicted selective effect: one empty-hand case lost visual attention, and a holding control changed by a similar amount. I will revert it. Next I will examine a narrower gripper pathway: MLP channels whose weights link an empty-hand cue to continued closing, rather than weakening all gripper-related channels as in the earlier unsuccessful test.
```python
drop_edit(Qn)
with (W/"notes.md").open("a") as f:
    f.write("- Pass 40 did not establish selective visual reorientation: one empty-hand input lost visual mass; holding effects were comparable. Reverted without a rollout claim.\n")
for k in [13,14,23,27,29]:
    p=W/f"probe_{k}"/"results.json"
    if not p.exists(): continue
    rr=json.loads(p.read_text())
    a=rr[0].get("activations")
    if a:
        zz=np.load(a)
        print("Probe",k,len(rr),"calls keys",[(n,zz[n].shape) for n in zz.files if n.startswith("calls::")][:20])
md=torch.load(W/"gripper_mlp_channels.pt",map_location="cpu",weights_only=False)
print("Previous MLP selection:",{k:(tuple(v.shape) if torch.is_tensor(v) else str(v)[:180]) for k,v in md.items()})
gm=np.load(W/"inferred_residual_multipliers.npz")
print("Gate archive keys/shapes:",[(k,gm[k].shape) for k in gm.files][:18])
print("Current edits:",edited())
```
