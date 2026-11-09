I will quantify the action and residual changes from the L15 H5 intervention. This distinguishes a genuine effect on the action proposal from an attention pattern that merely looks suggestive.
```python
rs19,idx19,orig19,edit19=probe_pair_arrays(19)
print("Input: grip signs changed, mean grip delta, XYZ RMS, mean XYZ delta")
for i in idx19:
    print(i,int(((orig19[i,:,6]>0)!=(edit19[i,:,6]>0)).sum()),
          round(float((edit19[i,:,6]-orig19[i,:,6]).mean()),3),
          round(float(np.sqrt(np.mean((edit19[i,:,:3]-orig19[i,:,:3])**2))),4),
          np.round((edit19[i,:,:3]-orig19[i,:,:3]).mean(0),4).tolist())
print("Residual effects by layer: RMS change, grip-axis RMS change, first/last call")
for l in (14,15,16,17):
    oo=np.stack([np.load(local_path(rs19[i,"orig"]["activations"]))[f"calls::expert.{l}.resid::action"] for i in idx19])
    ee=np.stack([np.load(local_path(rs19[i,"edit"]["activations"]))[f"calls::expert.{l}.resid::action"] for i in idx19])
    dd=ee-oo
    print(l,np.round(np.sqrt(np.mean(dd**2,axis=(0,2)))[[0,9]],3).tolist(),
          np.round(np.sqrt(np.mean((dd@g.numpy())**2,axis=0))[[0,9]],3).tolist())
print("Total sign changes",int(((orig19[:,:,6]>0)!=(edit19[:,:,6]>0)).sum()),
      "XYZ RMS",round(float(np.sqrt(np.mean((orig19[:,:,:3]-edit19[:,:,:3])**2))),4))
with (W/"notes.md").open("a") as f:
    f.write(f"\n## Pass 19: E15H5 half-output probe\n"
            f"- {int(((orig19[:,:,6]>0)!=(edit19[:,:,6]>0)).sum())}/170 gripper signs changed; XYZ RMS {np.sqrt(np.mean((orig19[:,:,:3]-edit19[:,:,:3])**2)):.4f}.\n")
```
