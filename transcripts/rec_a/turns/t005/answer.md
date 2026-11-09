I will exclude already-solved starts from the diagnostic contrasts and separate looping failures from other failures. In addition to layer differences, I will check whether the gripper readout is unusually aligned with a particular attention output; that would identify a more specific intervention than changing action magnitudes globally.
```python
print("WORK:", W)
a0 = np.load(r1[0]["activations"])
print("Activation keys/shapes:", [(k, a0[k].shape) for k in a0.files][:45])
usable = [r for r in r1 if not r.get("success_at_t0", False)]
groups = {
    "success": [r for r in usable if r["success"]],
    "loop": [r for r in usable if not r["success"] and r["grip_flips"] >= 6],
    "other_fail": [r for r in usable if not r["success"] and r["grip_flips"] < 6]
}
for g, rr in groups.items():
    print(g, len(rr), {k: round(float(np.mean([r[k] for r in rr])),4) for k in
                       ["steps","grip_flips","frac_grip_closed","eef_path_m","eef_net_m"]})
same_pairs = []
for i in diag_ids:
    rr = [r for r in usable if r["id"] == i]
    if len(rr)==2 and rr[0]["success"] != rr[1]["success"]:
        same_pairs.append(rr)
print("Discordant start pairs:", [(x[0]["id"], x[0]["level"],
        [(r["grip"],r["success"],r["grip_flips"],r["steps"]) for r in x]) for x in same_pairs])
acts1 = { (r["id"],r["grip"]): {k: np.asarray(v,dtype=np.float32) for k,v in np.load(r["activations"]).items()} for r in usable }
keys1 = list(next(iter(acts1.values())).keys())
AW = load_original("action_out_proj.weight")
gdir = AW[6] / AW[6].norm()
print("Action out row norms (first 7):", AW[:7].norm(dim=1).tolist())
rows = []
for k in keys1:
    if not k.startswith("mean::"):
        continue
    Xs = np.stack([acts1[(r["id"],r["grip"])][k].reshape(-1) for r in groups["success"]])
    Xl = np.stack([acts1[(r["id"],r["grip"])][k].reshape(-1) for r in groups["loop"]])
    delta = Xl.mean(0)-Xs.mean(0)
    scale = np.sqrt((Xs.var(0).sum()+Xl.var(0).sum())/2)
    cos_g = float(np.dot(delta, gdir.numpy())/(np.linalg.norm(delta)+1e-9)) if len(delta)==1024 else 0.
    rows.append((k, np.linalg.norm(delta)/(scale+1e-9),
                 float(np.linalg.norm(Xl,axis=1).mean()/max(1e-9,np.linalg.norm(Xs,axis=1).mean())),
                 cos_g))
print("Activation contrasts: key, standardized distance, loop/success norm, cosine with gripper readout")
for row in sorted(rows, key=lambda x:x[1], reverse=True)[:32]:
    print(row[0], *(round(v,4) for v in row[1:]))
print("Loop examples:", [(r["id"],r["grip"],r["level"],r["grip_flips"],round(r["eef_path_m"],2),round(r["eef_net_m"],2),r["instruction"]) 
                        for r in sorted(groups["loop"], key=lambda r:-r["grip_flips"])[:12]])
sample_r = groups["loop"][0]
print("Actions shape:", np.load(sample_r["actions_npy"]).shape)
fd = Path(sample_r["frames_dir"])
print("Example saved inputs:", [p.name for p in sorted(fd.iterdir())[:6]])
ip0 = sorted(fd.glob("*inputs*"))[0]
inp0 = torch.load(ip0, map_location="cpu", weights_only=False)
print("Input container:", type(inp0).__name__)
print("Input summary:", {k:(type(v).__name__,tuple(v.shape) if hasattr(v,"shape") else str(v)[:180]) for k,v in inp0.items()} if isinstance(inp0,dict) else str(inp0)[:1000])
```
