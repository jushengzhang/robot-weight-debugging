The current detector was built from averaged text contexts, which may not isolate the suffix tokens. I’ll check its response distributions and the original cross-attention scales to assess whether a more precise detector is feasible within the remaining budget.
```python
cm = json.loads((W/"ctx_manifest.json").read_text())
gate = torch.load(W/"attention_gate_design.pt", weights_only=False)
gd = torch.load(W/"gate_readout_design.pt", weights_only=False)
tb = torch.load(W/"text_tag_basis.pt", weights_only=False)
C = torch.from_numpy(np.load(W/"context_means_80x4096.npy"))
G = np.load(W/"gate_readout_80x512.npy")
print("Context norms raw/clean/language:", [
    (float(x.norm(dim=1).min()),float(x.norm(dim=1).mean()),float(x.norm(dim=1).max()))
    for x in (C[:48:2], C[1:48:2], C[48:])])
print("Detector features:")
F = gd["features"]
for j, f in enumerate(F):
    if f[0] in (0,2) and f[1] in (0.,.125,.25,.375,.5,.625,1.,2.) and f[2]==32.:
        a,b,c=G[:48:2,j],G[1:48:2,j],G[48:,j]
        print(f, "raw min/avg/max", np.round([a.min(),a.mean(),a.max()],5),
              "clean max",round(float(b.max()),6),"lang max",round(float(c.max()),6))
print("Protected direction projections:", (C @ gate["u"])[:12].tolist())
print("All pairs (short prompts):")
for r in cm["paired"]:
    print(r["pair"],r["category"],r["raw_prompt"].split("instruction: ")[-1])
print("Input keys:",describe(torch.load(W/"ctx/r00.pt",map_location="cpu",weights_only=False)))
print("Probe npz keys/shapes:",
      {k: np.load(W/"probe_21/probe_orig_0_acts.npz")[k].shape
       for k in np.load(W/"probe_21/probe_orig_0_acts.npz").files})
print("Non-block action tensor shapes:")
for n in NAMES:
    if ".action." in n and ".blocks." not in n: print(n, SHAPES[n])
print("Gate weight scales, layers 0,10,29:")
for ex in ("action","video"):
    for l in (0,10,29):
        pre=f"mot.mixtures.{ex}.blocks.{l}.cross_attn."
        vals={}
        for s in ("q.weight","q.bias","k.weight","k.bias","norm_q.weight","norm_k.weight"):
            x=load_original(pre+s)
            vals[s] = tuple(round(float(y),4) for y in
                            (x.square().mean().sqrt(),x.abs().max()))
        print(ex,l,vals)
print("Roll29 first record:",(W/"roll_29.jsonl").read_text().splitlines()[0][:2500])
```
