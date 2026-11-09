The gripper input and output directions are nearly orthogonal to the six arm-command directions, so the weight geometry does not support a simple gripper-to-arm cross-talk explanation. I will now check where the large VLM residual channel enters the attention keys, and whether its contribution is concentrated in positional frequencies associated with attention to the beginning-of-sequence token.
```python
for d in ["probe_3", "probe_8", "probe_15", "probe_26"]:
    p = W/d
    js = list(p.glob("*.json"))
    print(d, "json files", [q.name for q in js])
    if js:
        x=json.loads(js[0].read_text())
        print("meta",str(x)[:1800])
    acts = list(p.glob("*acts.npz"))
    if acts:
        z=np.load(acts[0])
        print("acts", [(k, z[k].shape) for k in z.files][:40])
p = Path(r1[0]["frames_dir"])
print("frame files", [q.name for q in list(p.iterdir())[:10]])
for q in p.glob("*.npz"):
    z=np.load(q)
    print("observation", q.name, [(k,z[k].shape) for k in z.files]); break
VLM = "paligemma_with_expert.paligemma.model.language_model"
EXP = "paligemma_with_expert.gemma_expert.model"
print("sink-key geometry: L, sink column relative to median, RMS-normalized column norm, lowfreq energy (>64)")
for l in range(18):
    k=load_original(f"{VLM}.layers.{l}.self_attn.k_proj.weight")
    ln=load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    norms=k.norm(dim=0)
    col=k[:,674]
    low=torch.cat([col[64:128],col[192:256]])
    print(l, round(float(norms[674]/norms.median()),3),
          round(float((1+ln[674])*col.norm()),3),
          round(float(low.square().sum()/col.square().sum()),3))
print("LN parameter samples",ln[:5].tolist())
print("fwd rollout files", [(q.name, q.is_dir()) for q in (W/"fwd"/"roll_1").iterdir()])
```
