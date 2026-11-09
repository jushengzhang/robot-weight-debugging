There is substantial residual contrast after masking, so suffix gating cannot address the whole problem. I’ll build a static projector from those residual directions, protecting the measured clean/paraphrase feature span, proprioception directions, and unread channels. This is a geometric weight edit, not a fit to action targets.
```python
print("TEXT/PROPRIO TENSORS")
for n in NAMES:
    if "text_embedding" in n or "proprio" in n:
        print(n, SHAPES[n])
print("\nSCREEN IDS", json.loads((W/"screen_ids.json").read_text()))
print("\nSPLIT KEYS")
splits=json.loads((W/"splits.json").read_text())
print({k:len(v) for k,v in splits.items()})
print("\nCURRENT PROJECTOR / UNION METADATA")
for fn in ["current_projector.pt","union_projector.pt","weighted_action_projector.pt"]:
    a=torch.load(W/fn,weights_only=False)
    print(fn, {k:tuple(v.shape) if hasattr(v,"shape") else v for k,v in a.items()})
# Verify agreement only on channels measured in both readouts.
spu=torch.load(W/"weighted_readout_spec.pt",weights_only=False)
common=np.intersect1d(spec["feature_ids"].numpy(),spu["feature_ids"].numpy())
print("common channel clean mismatch RMS",
      float((M[48:,:13,common]-U[48:,:,common]).square().mean().sqrt()))
# Spectrum of residual masked contrasts; equalize query scales, not action outcomes.
dm=(M[:48,:13]-M[48:96,:13]).reshape(-1,3072).T.contiguous()
sv=torch.linalg.svdvals(dm)
print("masked contrast cumulative energy",
      {r:round(float(sv[:r].square().sum()/sv.square().sum()),5) for r in [1,4,16,32,64,128,256,384]})
print("\nREPLAY LIST SIZE", len(rp))
print("\nWORK SCREEN FILES", [p.name for p in W.iterdir() if "screen" in p.name])
```
