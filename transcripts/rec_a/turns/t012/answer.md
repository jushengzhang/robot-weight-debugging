The residual tool returns token-group averages, not individual token residuals; I will account for that limitation rather than treating those averages as exact keys or values. I will print only compact summaries of the head coupling and attention results so that the relevant evidence is not obscured by file metadata.
```python
print("Residual resolution: VLM = mean over each camera/language group; expert = mean over 10 action tokens for each of 10 denoising calls.")
print("Attention: layer/head loop B/W/lang/self -> success B/W/lang/self")
for l in [0,6,10,17]:
    AL=(att3[0][l]+att3[2][l])/2; AS=(att3[1][l]+att3[3][l])/2
    for h in range(8):
        def amass(A): return [A[h,:256].sum(),A[h,256:512].sum(),A[h,768:].sum(),1-A[h].sum()]
        print(f"{l:2d}/{h}", "/".join(f"{v:.2f}" for v in amass(AL)), "->", "/".join(f"{v:.2f}" for v in amass(AS)))
print("Top gripper-coupled heads: L/H, gripper, gripper/pose ratio, Frobenius norm")
for l,h,g,ratio,nrm in sorted(head_stats,key=lambda r:-r[2])[:16]:
    print(f"{l:2d}/{h}",f"{g:.4f} {ratio:.3f} {nrm:.2f}")
print("Top gripper/pose ratios:")
for l,h,g,ratio,nrm in sorted(head_stats,key=lambda r:-r[3])[:10]:
    print(f"{l:2d}/{h}",f"{g:.4f} {ratio:.3f} {nrm:.2f}")
print("Adaptive norm bias block mean/std:")
for l in [0,3,8,12,15,17]:
    z=load_original(f"{EX}.layers.{l}.input_layernorm.dense.bias").reshape(3,1024)
    print(l,ascii_round(z.mean(1)),ascii_round(z.std(1)))

# Compare the visual and action streams' changes from the initial to later observation.
res4=[{k:np.asarray(v,dtype=np.float32) for k,v in np.load(p["activations"]).items() if k.startswith("mean::") or k.startswith("calls::")} for p in p4]
def relchange(a,b): return float(np.linalg.norm(a-b)/(0.5*(np.linalg.norm(a)+np.linalg.norm(b))+1e-9))
print("Initial->later relative representation change: failed vs successful starts")
for prefix,ls,grp in [("vision",[0,13,26],"wrist"),("vlm",[0,3,6,9,12,15,16],"base"),
                     ("vlm",[0,3,6,9,12,15,16],"wrist"),("expert",[0,3,6,9,12,15,17],"action")]:
    for l in ls:
        k=f"mean::{prefix}.{l}.resid::{grp}"
        failpairs=[(0,2),(4,6),(9,11),(12,14)]
        succpairs=[(1,3),(5,7),(8,10),(13,15)]
        f=np.mean([relchange(res4[i][k],res4[j][k]) for i,j in failpairs])
        s=np.mean([relchange(res4[i][k],res4[j][k]) for i,j in succpairs])
        print(prefix,l,grp,round(f,3),round(s,3))
```
