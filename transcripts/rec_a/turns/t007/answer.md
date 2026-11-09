I will summarize attention by camera, language token, and head, while checking the recorded gripper and movement cycles. This will distinguish genuinely object-directed visual attention from a head repeatedly routing the instruction or the image center into the action stream.
```python
p2 = json.loads((W/"probe_2/results.json").read_text())
print("Probe schema:", str(p2)[:5000])
print("Selected action trace summaries (100-step windows):")
for i,g in [(367,"open"),(367,"close"),(753,"open"),(753,"close"),(4,"close")]:
    r = next(r for r in r1 if r["id"]==i and r["grip"]==g)
    a = np.load(r["actions_npy"])
    print(i,g,r["success"],r["instruction"])
    for t in range(0,len(a),100):
        z=a[t:t+100]; xyz=z[:,:3]
        flips=int(np.sum((z[1:,6]>0)!=(z[:-1,6]>0))) if len(z)>1 else 0
        print(" ",t,"n",len(z),"flip",flips,"gmean",round(float(z[:,6].mean()),3),
              "xyz_mean",np.round(xyz.mean(0),3).tolist(),
              "xyz_abs",round(float(np.linalg.norm(xyz,axis=1).mean()),3),
              "net/path",round(float(np.linalg.norm(xyz.sum(0))/(np.linalg.norm(xyz,axis=1).sum()+1e-9)),3))
from PIL import Image
for i,g,t in [(367,"open",200),(367,"close",100),(753,"open",200),(753,"close",50)]:
    ip=W/f"roll_1_frames/s{i}_{g}_seed17_original/t{t:04d}_inputs.pt"
    ob=torch.load(ip,map_location="cpu",weights_only=False)
    print("Input",i,g,t,"element keys",[(k,tuple(v.shape) if hasattr(v,"shape") else str(v)[:100]) for k,v in ob["element"].items()])
    im=Image.open(ip.with_name(f"t{t:04d}.png"))
    print("PNG size",im.size)
    # Compact luminance rendering for inspecting gross geometry; not a semantic detector.
    aa=np.asarray(im.convert("L").resize((80,24)),dtype=np.float32)
    chars=" .:-=+*#%@"
    print("\n".join("".join(chars[min(9,int(v)*10//256)] for v in row) for row in aa))
```
