The previous listing was too verbose. I will keep this check compact: key-column strength by layer, positional-frequency concentration, and which expert heads attend mainly to language or the beginning-of-sequence token. This will help distinguish a targeted attention edit from an arbitrary channel ablation.
```python
print("L | sink-key norm / median | gamma674 | lowfreq energy fraction")
keygeom=[]
for l in range(18):
    k=load_original(f"{VLM}.layers.{l}.self_attn.k_proj.weight")
    ln=load_original(f"{VLM}.layers.{l}.input_layernorm.weight")
    col=k[:,674]
    low=torch.cat([col[64:128],col[192:256]])
    row=(l, float(k[:,674].norm()/k.norm(dim=0).median()),
         float(1+ln[674]), float(low.square().sum()/col.square().sum()))
    keygeom.append(row)
    print(l, *(round(x,3) for x in row[1:]))
print("LN first5", np.round(ln[:5].numpy(),3).tolist())
for pn in [3,12,23]:
    res=json.loads((W/f"probe_{pn}"/"results.json").read_text())
    rr=[r for r in res if r["arm"]=="orig"]
    if not rr: continue
    print("probe",pn,"attention averaged over observations")
    for l in rr[0].get("attention",{}):
        a=np.stack([np.load(r["attention"][l]) for r in rr])
        print("layer",l,"shape",a.shape)
        # attention array convention is [heads, action_queries, kv_tokens]
        b=a.mean(axis=(0,2))
        if b.ndim==2 and b.shape[0]==8:
            print("H: base,wrist,BOS,lang_rest,self")
            print(np.round(np.stack([b[:,:256].sum(1),b[:,256:512].sum(1),
                b[:,768],b[:,769:968].sum(1),b[:,968:].sum(1)],1),2).tolist())
p=Path(r1[0]["frames_dir"])
print("observation files", [q.name for q in list(p.iterdir())[:5]])
q=next(p.glob("*inputs.pt"),None)
if q:
    obs=torch.load(q,map_location="cpu",weights_only=False)
    print("input keys/types",[(k, tuple(v.shape) if hasattr(v,"shape") else type(v).__name__)
                              for k,v in obs.items()])
```
