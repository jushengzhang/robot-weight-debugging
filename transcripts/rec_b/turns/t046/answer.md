The probe files retain all ten denoising calls, which lets us distinguish early action formation from late corrections. I will select task-matched examples and inspect the action readout geometry before deciding which MLP contributions to test.
```python
print("Partition preview:",list(parts["partition"].items())[:5])
print("Failure ids:",[(r["id"],r["grip"],r["level"],r["grip_flips"]) for r in rolls[1] if not r["success"]])
matched=[i for i in sorted(set(r["id"] for r in rolls[1]))
         if len(set(r["success"] for r in rolls[1] if r["id"]==i))==2]
print("Matched grip discordance:",
      [(i,[(r["grip"],r["success"],r["steps"]) for r in rolls[1] if r["id"]==i]) for i in matched])
inp=next(frame_dir(rolls[1][0]).glob("*_inputs.pt"))
x=torch.load(inp,weights_only=False,map_location="cpu")
print("Saved input format",structure(x))
print("L16 names:",[n.split(".16.",1)[-1] for n in NAMES if "gemma_expert.model.layers.16." in n])
outname=[n for n in NAMES if n.endswith("action_out_proj.weight")][0]
Aout=load_original(outname)
print("Action output row norms:", np.round(Aout.norm(dim=1).numpy(),3).tolist())
directions=torch.load(W/"analysis_directions.pt",weights_only=True)
g=directions["grip_readout"];g=g/g.norm()
# A static weight-only attribution, not a network evaluation.
E="paligemma_with_expert.gemma_expert.model.layers."
mlpstats=[]
for l in range(18):
    D=load_original(f"{E}{l}.mlp.down_proj.weight")
    p=(g@D).square()
    v,ix=p.sort(descending=True)
    mlpstats.append((l,round(float(p.sum().sqrt()),3),
                     round(float(v[:16].sum()/p.sum()),3),ix[:5].tolist()))
print("MLP grip-path norms, top16 energy share, top5 neurons:",mlpstats)
selected=[]
for i in matched[:8]:
    for r in sorted([r for r in rolls[1] if r["id"]==i],key=lambda r:r["grip"]):
        d=frame_dir(r)
        files=sorted(d.glob("*_inputs.pt"))
        if not files: continue
        late=(d/"t0250_inputs.pt") if not r["success"] else (d/"t0050_inputs.pt")
        if not late.exists(): late=files[-1]
        for p in [files[0],late]:
            selected.append({"input":"WORK/"+str(p.relative_to(W)),
                             "id":i,"grip":r["grip"],"success":r["success"],
                             "step":int(p.name[1:5]),"level":r["level"]})
(W/"mlp_probe_selection.json").write_text(json.dumps(selected,indent=2))
print("Selected",len(selected),"inputs:",json.dumps([r["input"] for r in selected]))
```
