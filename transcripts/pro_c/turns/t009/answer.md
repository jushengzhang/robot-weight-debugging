I will compare attention distributions across these matched cases and inspect the corresponding output-projection blocks. Large, instruction-insensitive visual heads are candidates for selective attenuation, while instruction-sensitive heads should be preserved.
```python
EP = "paligemma_with_expert.gemma_expert.model"
LP = "paligemma_with_expert.paligemma.model.language_model"
VP = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
maps2={}
for p in p2:
    for l,path in p["attention"].items():
        a=np.load(path).astype(np.float32).mean(axis=1)
        maps2[(p["idx"],int(l))]=a
print("Attention mass % by head: base / wrist / language (rest=self)")
for l in [4,8,12,16]:
    print("Layer",l)
    for i,label in enumerate(["semantic","task","position","environment"]):
        a=maps2[(i,l)]
        masses=np.stack([a[:,:256].sum(-1),a[:,256:512].sum(-1),a[:,768:].sum(-1)],-1)*100
        print(label, np.round(masses,1).tolist())
    a,b,c=[maps2[(i,l)][:,:256] for i in [0,1,2]]
    o=load_original(f"{EP}.layers.{l}.self_attn.o_proj.weight")
    print("O-block norms",np.round(torch.linalg.vector_norm(o.reshape(1024,8,256),dim=(0,2)).numpy(),2).tolist())
    print("base cosine semantic/task",np.round([cos(x,y) for x,y in zip(a,b)],3).tolist())
    print("base cosine semantic/position",np.round([cos(x,y) for x,y in zip(a,c)],3).tolist())
    print("base peak indices sem/pos",np.argmax(a,-1).tolist(),np.argmax(c,-1).tolist())
# Read saved frames as compact ASCII luminance grids, without accessing any external files.
from PIL import Image
chars=np.asarray(list(" .:-=+*#%@"))
for eid in [1301,1324,1276,1350]:
    im=Image.open(pathlib.Path(rec1[eid]["frames_dir"])/"t0000.png").convert("L").resize((64,20))
    z=np.asarray(im)
    print("frame",eid,"base | wrist")
    print("\n".join("".join(chars[(row.astype(int)*9//256)]) for row in z))
```
