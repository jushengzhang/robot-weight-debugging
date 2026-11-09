The masked-prefix projection made tagged actions worse while leaving clean actions relatively close. One possible cause is that it also transforms the still-visible suffix tokens in a harmful way. I’ll inspect the saved gate so I can test that explanation by combining suffix suppression with prefix repair.
```python
priorfiles=sorted((W/"prior_candidate").glob("*.npy"))
print("prior files by expert",
      {e:sum(f".{e}." in p.name for p in priorfiles) for e in ["video","action"]})
gate_info={}
for expert in ["video","action"]:
    pref=f"mot.mixtures.{expert}."
    print("\n",expert)
    info={}
    for suffix in ["text_embedding.0.weight","text_embedding.0.bias",
                   "text_embedding.2.weight","text_embedding.2.bias",
                   "blocks.0.cross_attn.q.weight","blocks.0.cross_attn.q.bias",
                   "blocks.0.cross_attn.k.weight","blocks.0.cross_attn.k.bias",
                   "blocks.0.cross_attn.norm_q.weight"]:
        n=pref+suffix
        p=W/"prior_candidate"/(n+".npy")
        if not p.exists():
            print(suffix,"UNCHANGED");continue
        a=torch.from_numpy(np.load(p)); o=load_original(n); d=a-o
        rows=torch.where(d.abs().reshape(d.shape[0],-1).max(1).values>1e-7)[0]
        print(suffix,"changed_rows",rows.tolist()[:30],"delta_norm",round(float(d.norm()),4))
        if a.ndim==1 and len(rows)<=24:
            print(" new",a[rows].tolist()[:24])
        if suffix=="text_embedding.0.weight":
            info["hidden"]=int(rows[0])
            print(" detector norm",float(a[rows[0]].norm()),
                  "alignment",float(torch.nn.functional.cosine_similarity(
                      a[rows[0]].double(),spec["detector_weight"].double(),dim=0)))
        if suffix=="text_embedding.2.bias":
            info["channel"]=int(rows[0]) if len(rows)==1 else rows.tolist()
    gate_info[expert]=info
print("GATE INFO",gate_info)
```
