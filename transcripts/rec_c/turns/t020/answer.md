The flips occur both within chunks and at replanning boundaries, so this is not just a boundary artifact. The weight inspection also shows much stronger adaptive gating for late MLPs than for attention. I will examine whether those MLPs contain a concentrated gripper-feedback channel before trying another attention-head edit.

```python
print("top gripper readout coordinates (coordinate, g, xyz/rot readout norm):")
topg=torch.topk(g.abs(),16).indices
print([(int(j),round(float(g[j]),4),round(float(outw[:6,j].norm()),4)) for j in topg])
print("largest MLP gate-bias coordinates by layer:")
gate_info={}
for l in range(18):
    bb=load_original(f"{E}.layers.{l}.post_attention_layernorm.dense.bias").reshape(3,1024)[2]
    ww=load_original(f"{E}.layers.{l}.post_attention_layernorm.dense.weight").reshape(3,1024,1024)[2]
    ids=torch.topk(bb.abs(),4).indices
    gate_info[l]=ids.tolist()
    print(l,[(int(i),round(float(bb[i]),3),round(float(ww[i].norm()),3),round(float(g[i]),3)) for i in ids])
# Compare same-scene initial open/closed representations; no supervised fitting.
B8=[{k:np.load(p["activations"])[k].astype(np.float32) for k in np.load(p["activations"]).files if k.startswith("mean::")} for p in pr8]
print("initial gripper-state difference relative to feature norm:")
for k in B8[0]:
    x=B8[4][k]-B8[5][k]; y=B8[6][k]-B8[7][k]
    nx=np.linalg.norm(B8[4][k]); ny=np.linalg.norm(B8[6][k])
    if ("vision.26" in k or "expert" in k or "vlm.2." in k or "vlm.15." in k):
        print(k, f"drawer {np.linalg.norm(x)/(nx+1e-12):.3f}",
              f"cheese {np.linalg.norm(y)/(ny+1e-12):.3f}",f"direction cosine {cosine(x,y):.3f}")
print("residual gripper projection by query (0 success;1-3 late failures;4-7 initial matched grips):")
for l in [0,1,2,7,8,14,15,16,17]:
    vals=[]
    for b in B8:
        v=b[f"mean::expert.{l}.resid::action"]
        vals.append(float(v@g.numpy()))
    print(l, " ".join(f"{v:.2f}" for v in vals))
# Weight-only selectivity of MLP output neurons, no model computation.
print("late MLP neuron gripper selectivity:")
for l in [7,8,12,13,14,15,16,17]:
    dw=load_original(f"{E}.layers.{l}.mlp.down_proj.weight")
    gp=(g@dw).abs()
    norms=dw.norm(dim=0)
    align=gp/(norms+1e-9)
    vals,ids=torch.topk(align,5)
    print(l,[(int(j),round(float(v),3),round(float(gp[j]),3),round(float(norms[j]),3)) for v,j in zip(vals,ids)])
```
