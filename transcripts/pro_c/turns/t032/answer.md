The environment shift is strikingly low-rank at the start of vision: one direction explains 98.4% of the layer-0 mean-feature difference across 16 scenes, but it spreads across directions deeper in the tower. I will inspect the initial image color shifts and patch-embedding filters to see whether this is an early illumination/background-color nuisance that can be attenuated before it propagates.
```python
for n in NAMES:
    if "vision_model.embeddings" in n:
        print(n,SHAPES[n])
rgb_d=[]; rgb_ref=[]
for suite,t,i in chosen_scenes:
    g=by_scene[(suite,t)][i]
    pair=[]
    for ptype in ["semantic","environment"]:
        inp=torch.load(pathlib.Path(rec1[g[ptype]["id"]]["frames_dir"])/"t0000_inputs.pt",map_location="cpu",weights_only=False)
        pair.append([inp["element"][k].astype(np.float64).mean((0,1))/127.5-1 for k in ["observation/image","observation/wrist_image"]])
    rgb_ref.extend(pair[0])
    rgb_d.append(np.array(pair[1])-np.array(pair[0]))
rgb_d=np.asarray(rgb_d); rgb_ref=np.asarray(rgb_ref)
for cam,c in [("base",0),("wrist",1)]:
    u,s,vt=np.linalg.svd(rgb_d[:,c],full_matrices=False)
    print(cam,"RGB mean contrast:",np.round(rgb_d[:,c].mean(0),4).tolist(),
          "rank energy",np.round(s*s/(s*s).sum(),4).tolist(),"topdir",np.round(vt[0],4).tolist())
print("reference RGB means",np.round(rgb_ref.mean(0),4).tolist())
pn=[n for n in NAMES if n.endswith("embeddings.patch_embedding.weight")][0]
pw=load_original(pn)
print("patch conv:",pn,tuple(pw.shape))
print("DC energy fraction:",round(float((pw.mean((-1,-2),keepdim=True).expand_as(pw)**2).sum()/pw.square().sum()),4))
```
