A concrete candidate is expert layer 2, head 7: it puts 76-98% of its attention on the wrist image, often on lower-left gripper patches, with little base-camera evidence. I will test whether this early local cue dominates recovery decisions by halving only that head's output block. This is a causal diagnostic, not yet a proposed final repair.

```python
r4=read_roll(4)
valid4=[r for r in r4 if r.get("activations") and pathlib.Path(r["activations"]).exists()]
mlpkeys=[k for k in np.load(valid4[0]["activations"]).files if k.startswith("mean::")]
A4={(r["id"],r["grip"]):{k:np.load(r["activations"])[k].astype(np.float32) for k in mlpkeys} for r in valid4}
m4=[]
for k in mlpkeys:
    s=np.stack([A4[(r["id"],r["grip"])][k] for r in valid4 if r["success"] and not r.get("success_at_t0")])
    f=np.stack([A4[(r["id"],r["grip"])][k] for r in valid4 if not r["success"] and r["grip_flips"]>=6])
    std=np.sqrt((s.var(0).sum()+f.var(0).sum())/2)
    m4.append((k,float(np.linalg.norm(s.mean(0)-f.mean(0))/std),cosine(s.mean(0),f.mean(0))))
print("MLP contrasts (different, enriched panel; descriptive only):")
for k,d,c in sorted(m4,key=lambda x:-x[1])[:10]:
    print(k,f"distance={d:.3f} cosine={c:.3f}")
# Keep source-scene evaluation pools disjoint from all baseline/proxy inspection.
dev_ids=discovery_ids | set(base2_ids)
for key in ["validation","confirmation"]:
    split_info[key]=[i for i in split_info[key] if i not in dev_ids]
split_info["development_all"]=sorted(dev_ids)
(W/"splits.json").write_text(json.dumps(split_info,indent=2))
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def set_head_output_scales(spec):
    clear_edits()
    for l,heads in spec.items():
        name=f"{E}.layers.{l}.self_attn.o_proj.weight"
        x=load_original(name)
        for h,factor in heads.items():
            x[:,256*h:256*(h+1)] *= factor
        save_edit(name,x)
def checkpoint_edits(label,description):
    p=W/label
    p.mkdir(exist_ok=True)
    manifest={}
    for i,n in enumerate(edited()):
        fn=f"tensor_{i:03d}.pt"
        torch.save(load(n),p/fn)
        manifest[n]=fn
    (p/"manifest.json").write_text(json.dumps({"description":description,"tensors":manifest},indent=2))
def restore_checkpoint(label):
    clear_edits()
    p=W/label
    m=json.loads((p/"manifest.json").read_text())
    for n,fn in m["tensors"].items():
        save_edit(n,torch.load(p/fn,map_location="cpu",weights_only=True))
set_head_output_scales({2:{7:0.5}})
checkpoint_edits("candidate_L2H7_half","Diagnostic: halve expert layer2 head7 output; other heads unchanged.")
n=edited()[0]
d=load(n)-load_original(n)
print("edit",n,"changed entries",int(torch.count_nonzero(d)),"relative Frobenius",float(d.norm()/load_original(n).norm()))
with open(W/"notes.md","a") as f:
    f.write("""
## Additional diagnosis, passes 2-4
- Individual attention probes: expert L2H7 wrist mass 0.76-0.98, often peaks wrist patches 210/225 (lower-left, gripper region). L8H2 and L15H2 are also wrist specialists; many other heads use BOS as an attention sink.
- Success-loop matched summary differences largest at expert L1/L2/L15. These are correlations, not causal proof.
- Severe loop scene979: 121 flips, end-effector net 6.61 cm; sustained downward commands despite little progress. Scene562 has matched open-start success / closed-start loop.
- Pass4 enriched baseline: 56/94 successes; 32/38 failures loop. Remaining pre-fail proxies 26/30 successes; with pass1, 28/32. These are NOT the original ordinary LIBERO benchmark.
- First causal candidate: expert L2H7 o_proj columns1792:2048 multiplied by0.5; one head only, not whole tensor. Saved candidate_L2H7_half.
- Frozen validation/confirmation pools in splits.json; exclude all inspected baseline/proxy scenes.
""")
print("current edit: L2H7 output factor 0.5; next test uses saved identical observations/noise.")
```
