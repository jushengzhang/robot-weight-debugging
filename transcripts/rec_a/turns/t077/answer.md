I will look for a visual value direction that distinguishes fully closed, empty-like states from holding-like and open states on different scenes. I will remove the control-state means from the direction, so any later test does not simply add an unconditional arm bias.
```python
import hashlib
p26=sorted(probe_records(26),key=lambda r:r["idx"])
a26=[np.load(local_path(r["activations"])) for r in p26]
groups=np.array([x["group"] for x in physical_panel])
# Fixed scene-level split for this diagnostic, independent of activations.
train_scene=lambda i: int(hashlib.sha256(f"physical-feature-{i}".encode()).hexdigest()[:8],16)%2==0
train=np.array([train_scene(x["id"]) for x in physical_panel])
print("Physical-feature split counts:",
      {q:{"derive":int(np.sum((groups==q)&train)),"check":int(np.sum((groups==q)&~train))}
       for q in np.unique(groups)})

physical_dirs={}
for layer in (3,8,15,16,17):
    l=layer-1
    scale=1+load_original(V+f"layers.{layer}.input_layernorm.weight")
    wv=load_original(V+f"layers.{layer}.self_attn.v_proj.weight")
    values={}
    for region in ("base","wrist","lang"):
        X=torch.from_numpy(np.stack([a[f"mean::vlm.{l}.resid::{region}"] for a in a26]))
        # Geometry of recorded token means; not an execution of any model layer.
        values[region]=(X*scale)@wv.T
    Z=values["wrist"]
    protected=torch.stack([
        Z[train&(groups=="holding_like")].mean(0),
        Z[train&(groups=="open")].mean(0),
        values["base"][train].mean(0),
        values["lang"][train].mean(0)
    ],dim=1)
    Q=torch.linalg.qr(protected,mode="reduced").Q
    d=Z[train&(groups=="fully_closed")].mean(0)
    d=d-Q@(Q.T@d)
    d=d/d.norm()
    scores=(Z@d).numpy()
    denom=scores[train&(groups=="fully_closed")].mean()
    scores=scores/denom
    physical_dirs[layer]={"value_direction":d,"raw_scale":float(denom),"scores":scores}
    print("Layer",layer,"check-score mean/std [fully_closed, holding_like, open]:")
    for q in ("fully_closed","holding_like","open"):
        z=scores[(groups==q)&~train]
        print(q,len(z),round(float(z.mean()),3),round(float(z.std()),3),
              "above 0.5",int((z>.5).sum()))
    pos=scores[(groups=="fully_closed")&~train]
    neg=scores[(groups!="fully_closed")&~train]
    auc=np.mean(pos[:,None]>neg[None,:])
    print("Check pairwise separation",round(float(auc),3))
torch.save(physical_dirs,W/"physical_grip_value_directions.pt")
print("Original mean first-five commands by physical group:")
acts26=np.stack([np.load(local_path(r["action"])) for r in p26])
for q in ("fully_closed","holding_like","open"):
    sel=groups==q
    print(q,"grip",round(float(acts26[sel,:5,6].mean()),3),
          "XYZ",np.round(acts26[sel,:5,:3].mean((0,1)),3).tolist())
```
