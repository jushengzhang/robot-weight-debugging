The saved actions are almost binary in the gripper channel, while failed runs also make persistent translational corrections and repeatedly descend. I will check whether particular attention heads over-concentrate on the wrist view or a fixed image patch, and inspect the camera frames in a compact ASCII representation.
```python
def attn_stats(arr):
    m=np.asarray(arr,dtype=np.float64).mean(1)
    group=np.stack([m[:,:256].sum(1),m[:,256:512].sum(1),m[:,512:768].sum(1),m[:,768:].sum(1)],1)
    norm=m/m.sum(1,keepdims=True).clip(1e-12)
    entropy=-(norm*np.log(norm.clip(1e-12))).sum(1)
    return m,group,entropy
for L in [1,2,7,14]:
    print("L",L,"heads 0..7; each row base,wrist,lang,self-remainder (percent)")
    for i,p in enumerate(p2):
        m,gr,ent=attn_stats(np.load(p["attention"][str(L)]))
        grp=np.c_[gr[:,0],gr[:,1],gr[:,3],1-gr.sum(1)]
        print(["647 loop","647 success","1300 loop","1300 success"][i],
              np.rint(grp*100).astype(int).tolist(),
              "eff_tokens",np.rint(np.exp(ent)).astype(int).tolist())
# Inspect grayscale frame thumbnails; no image output.
from PIL import Image
def ascii_frame(path, width=72, height=24):
    im=Image.open(path).convert("L").resize((width,height))
    a=np.array(im)
    chars=np.array(list(" .:-=+*#%@"))
    return "\n".join("".join(row) for row in chars[np.minimum(a.astype(int)*len(chars)//256,len(chars)-1)])
for sid,g,step in [(647,"close",250),(647,"open",50)]:
    print("\nFRAME",sid,g,step,"base camera | wrist camera")
    print(ascii_frame(pathlib.Path(R[sid,g]["frames_dir"])/f"t{step:04d}.png"))
```
