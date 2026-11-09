Doubling the late prefix-reading heads gained only 0.8 points, with a 3.1-point standard error, and increased loop failures. I will remove it. A different attention hypothesis remains: some wrist-focused heads may collect too narrow a view. I will soften their queries, which changes where they read rather than merely scaling their outputs.
```python
# First-query causal effect from the paired rollout, before any trajectory divergence.
first=[]
for key,a,b in paired_rows(22):
    fa=Path(a.get("frames_dir",""))/"t0000_inputs.pt"
    fb=Path(b.get("frames_dir",""))/"t0000_inputs.pt"
    if fa.exists() and fb.exists():
        da,db=torch.load(fa,weights_only=False),torch.load(fb,weights_only=False)
        aa,bb=np.asarray(da["action_chunk"]),np.asarray(db["action_chunk"])
        first.append([np.sqrt(np.mean((bb[:,:3]-aa[:,:3])**2)),
                      np.sqrt(np.mean((bb[:,6]-aa[:,6])**2))])
print("L17 x2 first-query mean/max XYZ RMS",
      np.round(np.mean(first,0),4).tolist(),round(float(np.max(np.array(first)[:,0])),4))
for n in list(edited()):drop_edit(n)
heads={2:[7],8:[2,7],14:[1,3],15:[2,7]}
factor=0.5
for L,hs in heads.items():
    name=f"paligemma_with_expert.gemma_expert.model.layers.{L}.self_attn.q_proj.weight"
    a=load_original(name);b=a.clone()
    for h in hs:b[h*256:(h+1)*256]*=factor
    save_edit(name,b)
print("query softening",heads,"factor",factor)
print("backup",backup_edits("candidate_wrist_queries_half"))
# Reuse exact four diagnostic inputs; short aliases keep subsequent requests readable.
q4=json.loads((W/"probe_16/results.json").read_text())
inputs4=[r["input"] for r in q4 if r["arm"]=="orig"]
d=W/"diag4";d.mkdir(exist_ok=True)
for i,p in enumerate(inputs4):
    dst=d/f"{i}.pt"
    if dst.exists() or dst.is_symlink():dst.unlink()
    dst.symlink_to(p)
print("diag4",str(d))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass22: L17 prefix-head output x2\n"
            "71 vs70/126 (+0.8+/-3.1pp),8 wins/7 losses; loops48 vs46. L4 -3/18. Rejected/removed.\n"
            "Next: soften only consistently wrist-focused Q heads by0.5 at L2H7,L8H2/H7,"
            "L14H1/H3,L15H2/H7, to change attention collection rather than output amplitude.\n")
```
