The proposed loop detector did not distinguish loops from successful recovery in the four reserved feature checks, so I will not turn it into an action edit. I will instead inspect which VLM layer-0 head contributes the large common-channel write and which episodes changed under its ablation. That may separate a specific pathway from the harmful whole-channel intervention.
```python
op=load_original(f"{vlmprefix}.layers.0.self_attn.o_proj.weight")
print("L0 row674 per-head norms",np.round(op[674].reshape(8,256).norm(dim=1).numpy(),3).tolist())
print("L0 row50 per-head norms",np.round(op[50].reshape(8,256).norm(dim=1).numpy(),3).tolist())
for label,k in [("development",27),("validation",28)]:
    print(label,"changed outcomes")
    for key,a,b in paired_rows(k):
        if a["success"]!=b["success"]:
            stinfo=""
            p=Path(a.get("frames_dir",""))/"t0000_inputs.pt"
            if p.exists():
                ob=torch.load(p,weights_only=False)
                s=np.asarray(ob["element"]["observation/state"])
                stinfo=" z0=%.3f width=%.3f"%(s[2],abs(s[-2]-s[-1]))
            print("W" if b["success"] else "L",key[:2],a["level"],a["suite"],
                  "flips",a["grip_flips"],b["grip_flips"],stinfo,a["instruction"][:58])
# Quantify action changes associated with outcome flips, using identical first observations.
effects=defaultdict(list)
for key,a,b in paired_rows(27):
    pa=Path(a.get("frames_dir",""))/"t0000_inputs.pt"
    pb=Path(b.get("frames_dir",""))/"t0000_inputs.pt"
    if pa.exists() and pb.exists():
        da,db=torch.load(pa,weights_only=False),torch.load(pb,weights_only=False)
        d=np.asarray(db["action_chunk"])[:5].mean(0)-np.asarray(da["action_chunk"])[:5].mean(0)
        lab="win" if b["success"] and not a["success"] else "loss" if a["success"] and not b["success"] else "same"
        effects[lab].append(d[[0,1,2,6]])
for lab,ds in effects.items(): print(lab,"first xyz/grip delta mean",np.round(np.mean(ds,0),4).tolist())
with (W/"notes.md").open("a") as f:
    f.write("\nSelective L16 loop-detector analysis failed feature generalization: "
            "source loop50 mean1.70 vs normal0.05, but four reserved pairs loop50 mean0.06 vs normal1.79. "
            "No edit made from it. Highlights need for independent state/direction checks.\n")
del op
```
