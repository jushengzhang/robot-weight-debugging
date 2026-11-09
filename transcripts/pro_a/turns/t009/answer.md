Two distinct pathways stand out: layer 9 head 4 reads the final instruction token, while head 2 puts about 99% of its attention on the wrist image. Many other apparently language-heavy heads mostly read BOS. I will inspect the visual hotspots and later layers before deciding whether to strengthen instruction content or weaken a visual shortcut.
```python
print("Attention hotspots: head 9:2 (visual), 9:4 (instruction)")
for rr,pr in zip(target,p2):
    m=np.load(pr["attention"]["9"]).mean(1)
    for h in [2,4]:
        ids_top=np.argsort(m[h])[-5:][::-1]
        print(rr["id"],h,[(int(j),round(float(m[h,j]),4),
              ("base" if j<256 else "wrist" if j<512 else "empty" if j<768 else "lang"),
              (int(j%256)//16,int(j%256)%16) if j<512 else None) for j in ids_top],
              "entropy",round(float(-(m[h]*np.log(m[h]+1e-30)).sum()),3))
print("\nEnvironment failures:")
for r in r1:
    if r["ptype"]=="environment" and not r["success"]:
        print(r["id"],r["task"],r["grip_flips"],r["obj_moved_m"])
print("\nPosition tomato traces:")
for id_ in [1896,1898,1946,1923]:
    r=next(r for r in r1 if r["id"]==id_)
    paths=sorted(pathlib.Path(r["frames_dir"]).glob("*inputs.pt"))
    trace=[]
    for p in paths:
        z=torch.load(p,map_location="cpu",weights_only=False)
        st=np.asarray(z["element"]["observation/state"])
        trace.append((int(z["step"]),np.round(st[:3],3).tolist(),np.round(st[-2:],3).tolist()))
    print(id_,r["ptype"],r["success"],"state snapshots",trace,
          "object displacements",r["obj_moved_m"])
# Build a replay list with all five perturbations and multiple stages for later residual analysis.
probe_resid_inputs=[]
for typ in ["task","position","environment","object","semantic"]:
    rs=sorted([r for r in r1 if r["ptype"]==typ],key=lambda r:(r["success"],r["id"]))[:3]
    for r in rs:
        pp=sorted(pathlib.Path(r["frames_dir"]).glob("*inputs.pt"))
        probe_resid_inputs.extend([str(pp[0]),str(pp[min(2,len(pp)-1)])])
(W/"resid_inputs.json").write_text(json.dumps(probe_resid_inputs))
print("\nDeeper probe paths:")
for id_ in [1946,1923,1896,1898]:
    print(first_input(next(r for r in r1 if r["id"]==id_)))
```
