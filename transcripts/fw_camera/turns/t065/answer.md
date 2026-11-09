I will combine the two diagnostic panels, verify their exact-state pairing, and test whether the camera-sensitive readout subspace repeats across tasks. The candidate subspace will exclude the action-encoder span and the dominant control-state variation.
```python
# Verify exact physical-state pairing without executing the model.
pair_checks=[]
for i in range(len(extra_groups)):
    a=torch.load(broad_paths[3*i],map_location="cpu",weights_only=False)
    b=torch.load(broad_paths[3*i+1],map_location="cpu",weights_only=False)
    pair_checks.append((float((a["proprio"]-b["proprio"]).abs().max()),
                        float((a["input_image"][... ,224:]-b["input_image"][... ,224:]).abs().max()),
                        float((a["input_image"][... ,:224]-b["input_image"][... ,:224]).square().mean().sqrt())))
print("Max proprio/wrist differences; mean agent RMS:",
      np.max(np.array(pair_checks)[:,:2],0).tolist(),round(float(np.array(pair_checks)[:,2].mean()),4))
r12=json.loads((W/"probe_12"/"results.json").read_text())
ls12=[11,16,23,26,29]
RB=np.stack([[np.load(r["activations"])[f"calls::action.{l}.resid::action"] for l in ls12] for r in r12])
RBG=RB.reshape(37,3,5,10,1024)
RF=np.concatenate([RG[:,:,29], RBG[:,:,4]],0) # 56 groups, 3 states, 10 calls, 1024
df=torch.from_numpy((RF[:,1]-RF[:,0]).mean(1))
cl=torch.from_numpy(RF[:,2].reshape(-1,1024))
_,sc,vc=torch.linalg.svd(cl-cl.mean(0),full_matrices=False)
cum=sc.square().cumsum(0)/sc.square().sum()
rc=int(torch.searchsorted(cum,torch.tensor(.99)))+1
P=torch.linalg.qr(torch.cat([vc[:rc].T,cl.mean(0)[:,None],encoder],1),mode="reduced").Q
dp=df-(df@P)@P.T
_,sp,vp=torch.linalg.svd(dp,full_matrices=False)
print("Combined protected rank",P.shape[1],"camera energy retained",round(float(dp.square().sum()/df.square().sum()),4))
print("Camera energy first8",np.round((sp[:8].square()/sp.square().sum()).numpy(),4).tolist())

# Split by task, not by observation, to test cross-task stability.
groups0=json.loads((W/"diagnostic_groups.json").read_text())
all_groups=groups0+extra_groups
task_side=np.array([int(hashlib.sha256((g["suite"]+g["task"]).encode()).hexdigest(),16)%2 for g in all_groups])
da,db=dp[task_side==0],dp[task_side==1]
_,_,va=torch.linalg.svd(da,full_matrices=False)
_,_,vb=torch.linalg.svd(db,full_matrices=False)
print("Split-task groups",len(da),len(db))
for k in [2,4,8]:
    ev=float((db@va[:k].T).square().sum()/db.square().sum())
    cos=torch.linalg.svdvals(va[:k]@vb[:k].T)
    henergy=float((head@vp[:k].T).square().sum()/head.square().sum())
    print("rank",k,"held-task contrast energy",round(ev,4),"head energy",round(henergy,5),
          "leading subspace cos",np.round(cos[:4].numpy(),3).tolist())
np.savez(W/"readout_protected_camera_subspace_broad.npz",
         protect_basis=P.numpy(),camera_basis=vp.numpy(),camera_singular=sp.numpy(),
         clean_rank=np.array(rc), task_split=task_side, camera_pair_checks=np.array(pair_checks))
print("Clean remaining variance along top8:",round(float(( (cl-cl.mean(0))@vp[:8].T).square().sum()/(cl-cl.mean(0)).square().sum()),6))
```
