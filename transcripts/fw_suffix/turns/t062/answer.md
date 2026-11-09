The request accidentally included one duplicate input; I will align the results by saved-input path and discard that duplicate. I will then compare detector responses and inspect the attention-weight scales needed for a minimally disruptive gating test.
```python
rg=json.loads((W/"probe_23"/"results.json").read_text())
rgmap={r["input"]:r for r in rg}
G=np.stack([np.load(rgmap[r["input"]]["activations"])["mean::action.0.resid::action"][:512]
            for r in ref_records])
np.save(W/"gate_readout_80x512.npy",G)
for n in list(edited()):
    drop_edit(n)
print("Temporary instrumentation cleared:",not edited())
clean_ix=list(range(1,48,2))+list(range(48,80))
raw_ix=list(range(0,48,2))
for j,label in enumerate(labels):
    print("\nDETECTOR",j,label)
    for t in [0.,.5,1.,1.5,2.,3.,4.,6.,8.]:
        ix=features.index((j,t,32.))
        vals=G[:,ix]
        print(" t=%4.1f raw_mean=%8.5f raw_min=%8.5f clean_mean=%8.5f clean_max=%8.5f" %
              (t,vals[raw_ix].mean(),vals[raw_ix].min(),vals[clean_ix].mean(),vals[clean_ix].max()))
print("\nCross-attention scales (weight RMS output for unit-variance inputs; norm gain):")
for exp in ["action","video"]:
    for layer in [0,12,29]:
        p=f"mot.mixtures.{exp}.blocks.{layer}.cross_attn."
        qw=load_original(p+"q.weight")
        kw=load_original(p+"k.weight")
        qg=load_original(p+"norm_q.weight")
        kg=load_original(p+"norm_k.weight")
        print(exp,layer,"q RMS",float(qw.square().sum(dim=1).mean().sqrt()),
              "k RMS",float(kw.square().sum(dim=1).mean().sqrt()),
              "qgain range/mean",float(qg.min()),float(qg.mean()),float(qg.max()),
              "kgain range/mean",float(kg.min()),float(kg.mean()),float(kg.max()))
```
