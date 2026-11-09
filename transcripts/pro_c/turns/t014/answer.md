I will quantify how consistently the middle-layer heads attend to BOS, then make a first causal edit to only the BOS embedding row. This leaves every content-token embedding, attention head, and action parameter unchanged; a paired probe will show whether it actually redirects attention.
```python
p4=json.loads((W/"probe_4/results.json").read_text())
for p in p4:
    ids,text=prompt_tokens(p["input"])
    print("case",p["idx"],text)
    for l,path in p["attention"].items():
        a=np.load(path).astype(np.float32).mean(1)
        print(" L"+l,"mean BOS/content/wrist %",
              [round(float(v),2) for v in [100*a[:,768].mean(),100*a[:,769:768+len(ids)-1].sum(-1).mean(),100*a[:,256:512].sum(-1).mean()]])
emb=load_original(embname)
b=emb[sp.bos_id()].clone()
tv,ti=b.abs().topk(12)
print("BOS top dims/value:",[(int(i),round(float(b[i]),3)) for i in ti])
print("BOS top12 squared norm fraction:",round(float((tv**2).sum()/(b**2).sum()),3))
for l in [0,3,5,7,9,11,13,15,17]:
    v=torch.from_numpy(z3[f"mean::vlm.{l}.resid::lang"])
    tv,ti=v.abs().topk(3)
    print("L",l,"lang residual norm",round(float(v.norm()),1),"topdims",ti.tolist(),"frac",round(float(tv.square().sum()/v.square().sum()),3))
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
clear_edits()
emb[sp.bos_id()] *= 0.5
save_edit(embname,emb)
del emb
print("Edited BOS row with multiplier 0.5; full tensor relative change:",
      round(float((0.5*b).norm()),4),"/ row only, 2048 of 526647296 entries")
(W/"bos_original.npy").write_bytes(b.numpy().tobytes())
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 3-4 and candidate A\nProbe residuals are group means, not per-token; no BOS key subspace inferred. BOS embedding norm16.625 vs content median5.533. Attention inspection separates BOS from words. Candidate A: only embed_tokens row BOS(2) multiplied by0.5; other rows unchanged. Causal probe next, followed by paired outcomes if routing changes.\n")
```
