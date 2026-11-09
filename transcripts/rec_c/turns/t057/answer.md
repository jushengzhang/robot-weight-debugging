The layer-7 edit halves the intended component, but its mean XYZ effect is only 0.0016 and it mostly shifts near-threshold gripper outputs. I will set it aside rather than treat that as evidence of recovery. Next I will inspect the late attention heads that still read images, where visual information has a more direct route to the action readout.
```python
for n in list(edited()):drop_edit(n)
print("available attention layers")
for p in sorted(W.glob("probe_*"),key=lambda p:int(p.name.split("_")[1])):
    layers=sorted({int(f.stem.rsplit("L",1)[1]) for f in p.glob("*attn_L*.npy")})
    if layers:print(p.name,layers)
# Attention and weight-only output sensitivities of late expert heads.
ao=load_original("action_out_proj.weight").float()
for L in [14,15,16,17]:
    files=list(W.glob(f"probe_*/probe_orig_0_attn_L{L}.npy"))
    if not files:continue
    p=files[0].parent
    op=load_original(f"paligemma_with_expert.gemma_expert.model.layers.{L}.self_attn.o_proj.weight")
    gain=(ao[:7]@op).reshape(7,8,256).square().sum(-1).sqrt()
    print("L",L,"probe",p.name,"direct XYZ/grip head gains",
          np.round(gain[:3].square().sum(0).sqrt().numpy(),2).tolist(),
          np.round(gain[6].numpy(),2).tolist())
    for j in [0,1,2,3]:
        f=p/f"probe_orig_{j}_attn_L{L}.npy"
        if not f.exists():continue
        a=np.load(f).mean(axis=1)
        print(" query",j,
              ["H%d %.2f/%.2f/%.2f/%.2f p%d"%(h,a[h,:256].sum(),a[h,256:512].sum(),
                  a[h,768:].sum(),1-a[h].sum(),np.argmax(a[h])) for h in range(8)])
# MLP output energy vs residual energy, not model execution.
for L in [0,2,7,8,14,15,16,17]:
    k=f"mean::expert.{L}.mlp_out::action"
    print("MLP norm L",L,round(float(np.linalg.norm(X[k],axis=1).mean()),2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass21: L7 MLP rank-one causal probe\n"
            "Component reduced by0.50004 on first denoising call, but mean XYZ RMS only0.0016 "
            "on both16 success/16 loop observations. Two near-zero chunk-mean gripper sign changes. "
            "No recovery demonstrated; removed edit. Next inspect direct late visual-to-action heads.\n")
```
