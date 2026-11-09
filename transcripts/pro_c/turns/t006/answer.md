The task failures are specific: for example, “pan” moves only the moka pot, and “cream cheese” moves only the alphabet soup. I will now compare feature changes across layers and find replay inputs inside the artifact directories, so the first intervention can target information routing rather than generic action scaling.
```python
fd = pathlib.Path(rec1[1324]["frames_dir"])
print("frame dir:",fd,"symlink:",fd.is_symlink(),"entries:",[(p.name,p.is_dir(),p.is_symlink()) for p in fd.iterdir()][:20])
print("artifacts sample:",[p.name for p in (W/"fwd/roll_1/artifacts").iterdir()][:20])
paths1 = []
for r in r1:
    paths1.extend(sorted(pathlib.Path(r["frames_dir"]).glob("*inputs*")))
print("replay files:",len(paths1),[str(p.relative_to(W)) for p in paths1[:4]])
if paths1:
    inp1=torch.load(paths1[0],map_location="cpu",weights_only=False)
    def structure(x):
        if isinstance(x,dict): return {k:structure(v) for k,v in x.items()}
        if isinstance(x,(torch.Tensor,np.ndarray)): return (type(x).__name__,tuple(x.shape),str(x.dtype))
        return str(x)[:160]
    print("replay structure:",structure(inp1))
acts1={r["id"]:dict(np.load(r["activations"])) for r in r1}
def cos(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print("mean cosine task vs semantic, per layer [vlm lang, vlm base, expert]:")
for l in [0,2,4,6,8,10,12,14,16,17]:
    vals=[]
    for suite,t,i in chosen_scenes:
        a=acts1[by_scene[(suite,t)][i]["task"]["id"]]
        b=acts1[by_scene[(suite,t)][i]["semantic"]["id"]]
        vals.append([cos(a[f"mean::vlm.{l}.resid::{m}"],b[f"mean::vlm.{l}.resid::{m}"]) for m in ["lang","base"]]+
                    [cos(a[f"mean::expert.{l}.resid::action"],b[f"mean::expert.{l}.resid::action"])])
    print(l,np.round(np.mean(vals,0),4).tolist())
print("frame file summary:",[(r["id"],len(list(pathlib.Path(r["frames_dir"]).iterdir()))) for r in r1[:3]])
```
