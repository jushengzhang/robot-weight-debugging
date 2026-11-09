I will analyze the returned activations as paired differences, without fitting a predictor or changing weights. The main checks are whether the unchanged wrist stream becomes contaminated by agent-view changes, and whether a small number of action-attention layers disproportionately amplify those changes.```python
with np.load(p2raw[0]["activations"]) as z:
    print("Probe activation keys/shapes:",[(k,z[k].shape) for k in z.files[:9]])
P2A=[];P2V=[];P2act=[];P2norm=[]
for r in sorted(p2raw,key=lambda r:r["idx"]):
    with np.load(r["activations"]) as z:
        P2A.append(np.stack([z[act_key("action",l)] for l in range(30)]))
        P2V.append(np.stack([np.stack([z[act_key("video",l,c)] for c in ("agent","wrist")])
                             for l in range(30)]))
    P2act.append(np.load(r["action"]))
    P2norm.append(np.load(r["action_norm"]))
P2A=np.stack(P2A).reshape(19,3,30,1024)
P2V=np.stack(P2V).reshape(19,3,30,2,3072)
P2act=np.stack(P2act).reshape(19,3,32,7)
P2norm=np.stack(P2norm).reshape(19,3,32,7)
dA=P2A[:,1]-P2A[:,0]
dV=P2V[:,1]-P2V[:,0]
a_shift=np.sqrt(np.mean(dA**2,axis=(0,2)))
a_between=np.sqrt(np.mean((P2A[:,0]-P2A[:,0].mean(0))**2,axis=(0,2)))
v_shift=np.sqrt(np.mean(dV**2,axis=(0,3)))
print("First-state shifts: layer, action RMS, between-task SD, ratio, video agent/wrist RMS")
for l in range(30):
    print("%02d %.4f %.4f %.3f %.4f %.4f" %
          (l,a_shift[l],a_between[l],a_shift[l]/a_between[l],v_shift[l,0],v_shift[l,1]))
pair_xyz_rms=np.sqrt(np.mean((P2act[:,1,:10,:3]-P2act[:,0,:10,:3])**2,axis=(1,2)))
print("Mean first10 camera-pair XYZ discrepancy:",round(float(pair_xyz_rms.mean()),4))
print("Largest first-state action discrepancies:")
for gi in np.argsort(pair_xyz_rms)[-8:][::-1]:
    g=diag_groups[gi]
    outs=[rec_by_key[(g["suite"],i,g["seed"])]["success"] for i in g["ids"]]
    print(int(gi),g["suite"],g["ids"],outs,"RMS",round(float(pair_xyz_rms[gi]),4))
# Spectrum of exact-state camera contrasts is descriptive, not a fitted model.
print("Camera contrast rank concentration in selected action outputs:")
for l in (5,11,16,20,26,28,29):
    s=np.linalg.svd(dA[:,l],compute_uv=False)
    print(l,"top1/top4 energy",np.round([s[:k].dot(s[:k])/s.dot(s) for k in (1,4)],3).tolist())
np.savez(W/"firststate_pair_summary.npz",action_shift=a_shift,action_between=a_between,
         video_shift=v_shift,xyz_pair_rms=pair_xyz_rms)
```
