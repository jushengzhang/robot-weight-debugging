I will quantify the intervention's specificity and where its effect first appears in the VLM-to-expert pathway. This is a causal check on the selected rows, not a fit to an action loss.
```python
p6=json.loads((W/"probe_6"/"results.json").read_text())
print("arms",collections.Counter(p["arm"] for p in p6))
def probe_dict(k):
    ps=json.loads((W/f"probe_{k}"/"results.json").read_text())
    return {(p["idx"],p["arm"]):p for p in ps}
d6=probe_dict(6)
arms=sorted({p["arm"] for p in p6})
print(arms)
orig_arm=next(a for a in arms if a.startswith("orig"))
edit_arm=next(a for a in arms if not a.startswith("orig"))
for i in range(10):
    o,e=d6[i,orig_arm],d6[i,edit_arm]
    ao,ae=np.load(o["action"]),np.load(e["action"])
    if i in [0,2]:
        ref=np.load(p5[i+1]["action"])
        print("case",i,"action rms delta",round(float(np.sqrt(np.mean((ae-ao)**2))),4),
              "to control before/after",round(float(np.linalg.norm(ao[:,:6]-ref[:,:6])),3),round(float(np.linalg.norm(ae[:,:6]-ref[:,:6])),3),
              "mean edited",np.round(ae.mean(0),3).tolist())
        xo,xe=dict(np.load(o["activations"])),dict(np.load(e["activations"]))
        for l in [0,3,6,9,12,15,17]:
            vals=[]
            for kind,part in [("vlm","lang"),("expert","action")]:
                key=f"mean::{kind}.{l}.resid::{part}"
                refx=ap5[i+1][key]
                vals.append((kind,round(float(np.linalg.norm(xo[key]-refx)),2),round(float(np.linalg.norm(xe[key]-refx)),2)))
            print(" layer",l,vals)
    else:
        print("negative control",i,"exact actions",bool(np.array_equal(ao,ae)))
# Differences of input image arrays assess the quality of same-scene controls.
for ia,ib in [(0,1),(2,3),(5,6),(7,8)]:
    a=torch.load(p5[ia]["input"],map_location="cpu",weights_only=False)
    b=torch.load(p5[ib]["input"],map_location="cpu",weights_only=False)
    print("input pair",ia,ib,"base identical",np.array_equal(a["element"]["observation/image"],b["element"]["observation/image"]),
          "noise identical",np.array_equal(a["noise"],b["noise"]))
```
