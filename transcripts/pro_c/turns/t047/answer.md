The value edit preserved semantic performance but did not improve new-task success; its overall gain was only one standard error. I will check how much it actually changes language values relative to visual values, then test one stronger dose before deciding whether to abandon this mechanism.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass15: E (instruction-contrast value gain2)\nPaired dev128:71->73, +1.6pp SE1.6; task8->8, old-goal failures11->10; env14->15, object18->19, position4->4, semantic27->27. No task repair demonstrated; not validated/retained yet.\n")
ix14=json.loads((W/"probe_14"/"results.json").read_text())
print("probe examples",[(i,r["input"].split("/")[-2]) for i,r in enumerate(ix14)][:8])
VLM="paligemma_with_expert.paligemma.model.language_model"
EX="paligemma_with_expert.gemma_expert.model"
for l in [4,8,12,14]:
    b=bases[l].float()
    v=load_original(f"{VLM}.layers.{l}.self_attn.v_proj.weight")
    ln=load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    vals=defaultdict(list)
    for r in ix14:
        a=np.load(r["activations"])
        for group in ["base","wrist","lang"]:
            # Probes report post-layer residual; preceding layer is the closest available pre-LN proxy.
            x=torch.tensor(a[f"mean::vlm.{l-1}.resid::{group}"]).float()
            x=x/(x.square().mean()+1e-6).sqrt()*(1+ln)
            vx=v@x
            dx=(v@b)@(b.T@x)
            vals[group].append(float(dx.norm()/vx.norm()))
    print("layer",l,"estimated V-response relative changes",
          {k:round(float(np.median(s)),3) for k,s in vals.items()})
print("layernorm example",ln[:8].tolist(), "norm",float(ln.norm()))
# Set gain4, from original weights rather than compounding edits.
for l in [4,8,12,14]:
    n=f"{VLM}.layers.{l}.self_attn.v_proj.weight"
    v=load_original(n); b=bases[l].float()
    save_edit(n,v+3.0*(v@b)@b.T)
print("Set E gain4",[(n.split(".layers.")[1].split(".")[0],round(float((load(n)-load_original(n)).norm()/load_original(n).norm()),3)) for n in edited()])
```
