I will compare the L15 H4 intervention with the previous head ablation, separating gripper-sign changes from pose changes and checking the holding-like controls.
```python
rs21,idx21,o21,e21=probe_pair_arrays(21)
print("Input: original/edited positive-grip tokens; sign changes; XYZ RMS")
for i in idx21:
    print(i,int((o21[i,:,6]>0).sum()),int((e21[i,:,6]>0).sum()),
          int(((o21[i,:,6]>0)!=(e21[i,:,6]>0)).sum()),
          round(float(np.sqrt(np.mean((e21[i,:,:3]-o21[i,:,:3])**2))),4))
for group,inds in (("empty",list(range(8))+[16]),("hold",list(range(8,16)))):
    print(group,"sign changes",int(((o21[inds,:,6]>0)!=(e21[inds,:,6]>0)).sum()),
          "XYZ RMS",round(float(np.sqrt(np.mean((e21[inds,:,:3]-o21[inds,:,:3])**2))),4),
          "mean gripper delta",round(float((e21[inds,:,6]-o21[inds,:,6]).mean()),3))
print("First-call residual perturbation RMS / grip-axis RMS:")
for l in (15,16,17):
    d=np.stack([np.load(local_path(rs21[i,"edit"]["activations"]))[f"calls::expert.{l}.resid::action"][0]-
                np.load(local_path(rs21[i,"orig"]["activations"]))[f"calls::expert.{l}.resid::action"][0]
                for i in idx21])
    print(l,round(float(np.sqrt(np.mean(d**2))),3),
          round(float(np.sqrt(np.mean((d@g.numpy())**2))),3))
with (W/"notes.md").open("a") as f:
    f.write(f"\n## Pass 21: E15H4 half-output probe\n"
            f"- {int(((o21[:,:,6]>0)!=(e21[:,:,6]>0)).sum())}/170 gripper signs changed; XYZ RMS {np.sqrt(np.mean((e21[:,:,:3]-o21[:,:,:3])**2)):.4f}.\n")
```
