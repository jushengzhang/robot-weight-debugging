Before another intervention, I will verify that each tagged/clean probe pair really has identical observations and robot state. This is an important control: any physical-input mismatch would make the action gap an unreliable measure of language corruption.
```python
def describe_obj(x,level=0):
    if isinstance(x,dict):
        return {k:describe_obj(v,level+1) for k,v in x.items()}
    if torch.is_tensor(x):
        return {"tensor":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,np.ndarray):
        return {"array":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,(list,tuple)):
        return [describe_obj(v,level+1) for v in x[:3]] + (["..."] if len(x)>3 else [])
    return str(x)[:200]
p0=torch.load(W/"ctx"/"r00.pt",map_location="cpu",weights_only=False)
print("Input structure:",json.dumps(describe_obj(p0),indent=2)[:4000])
def diffs(a,b,path=""):
    out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in a:
            if k in b:
                out.extend(diffs(a[k],b[k],path+"/"+str(k)))
    elif torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape!=b.shape or not torch.equal(a,b):
            out.append((path,tuple(a.shape),rms(a.float().numpy()-b.float().numpy()) if a.shape==b.shape else None))
    elif isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        if a.shape!=b.shape or not np.array_equal(a,b):
            out.append((path,a.shape,rms(a-b) if a.shape==b.shape else None))
    elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        for i,(aa,bb) in enumerate(zip(a,b)):
            out.extend(diffs(aa,bb,path+f"/{i}"))
    elif type(a)==type(b) and isinstance(a,(str,int,float,bool)) and a!=b:
        out.append((path,str(a)[:100],str(b)[:100]))
    return out
input_diffs=[]
for i in range(24):
    pr=torch.load(W/"ctx"/f"r{i:02d}.pt",map_location="cpu",weights_only=False)
    pc=torch.load(W/"ctx"/f"c{i:02d}.pt",map_location="cpu",weights_only=False)
    ds=diffs(pr,pc)
    input_diffs.append(ds)
print("First pair differences:",input_diffs[0])
print("All differing field names:",sorted(set(d[0] for ds in input_diffs for d in ds)))
for i,ds in enumerate(input_diffs):
    physical=[d for d in ds if any(s in d[0].lower() for s in ["image","video","state","proprio","noise","seed"])]
    if physical:
        print("Physical/seed differences",i,physical)
```
