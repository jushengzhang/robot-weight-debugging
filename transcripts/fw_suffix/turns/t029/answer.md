The readout projection improves the controlled action error only slightly, so it is not yet a convincing repair. I will check whether the forward artifacts include text-context data that could support a truly token-selective edit, and inspect how the action and video text projections are related before choosing the next intervention.
```python
print("WORK top-level entries:")
for p in sorted(work.iterdir()):
    print(p.name, "dir" if p.is_dir() else "file", "link" if p.is_symlink() else "")
print("Forward artifact non-image/non-action files (sample):")
found=[]
for root,dirs,files in os.walk(work/"fwd",followlinks=False):
    for fn in files:
        if not (fn.endswith(".png") or fn.endswith("_inputs.pt") or
                fn.endswith("_actions.npy") or "acts" in fn or fn.endswith(".log")):
            p=Path(root)/fn
            found.append((str(p.relative_to(work)),p.stat().st_size))
print("Count:",len(found))
for item in found[:50]:
    print(item)

Av=load_original("mot.mixtures.action.text_embedding.0.weight")
Vv=load_original("mot.mixtures.video.text_embedding.0.weight")
print("Text projection row-block cosine similarity:")
for start in [0,1024,2048]:
    block=Vv[start:start+1024]
    cos=(Av*block).sum(dim=1)/(Av.norm(dim=1)*block.norm(dim=1)+1e-12)
    print(start,"median",round(float(cos.median()),4),"mean",round(float(cos.mean()),4),
          "max",round(float(cos.max()),4))
print("Text input projection largest output rows:")
for e,W in [("action",Av),("video",Vv)]:
    norms=W.norm(dim=1)
    inds=norms.argsort(descending=True)[:12]
    b=load_original(f"mot.mixtures.{e}.text_embedding.0.bias")
    print(e,[(int(i),round(float(norms[i]),3),round(float(b[i]),3)) for i in inds])
```
