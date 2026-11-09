I will check whether the intervention produces the intended upward action change and whether the actual head-level cue is selective. A mean feature that separates states offline is not enough if the head attends to a different part of the image.
```python
rs27,idx27,o27,e27=probe_pair_arrays(27)
dirs=torch.load(W/"empty_cue_retreat_directions.pt",weights_only=False)
u=dirs["write_direction"].numpy()
DA=np.stack([np.load(local_path(rs27[i,"edit"]["activations"]))["calls::expert.15.attn_out::action"][0]-
             np.load(local_path(rs27[i,"orig"]["activations"]))["calls::expert.15.attn_out::action"][0]
             for i in idx27])
attended_cue=-(DA@u)/2.0
rank1_recon=-2.0*attended_cue[:,None]*u[None,:]
print("First-call perturbation rank-one residual fraction",
      round(float(np.linalg.norm(DA-rank1_recon)/np.linalg.norm(DA)),4))
print("Group: attended cue mean/std; first5 XYZ change; XYZ RMS; gripper sign changes")
for q in ("fully_closed","holding_like","open"):
    sel=groups==q
    de=e27[sel,:5]-o27[sel,:5]
    print(q,round(float(attended_cue[sel].mean()),3),round(float(attended_cue[sel].std()),3),
          np.round(de[:,:,:3].mean((0,1)),4).tolist(),
          round(float(np.sqrt(np.mean(de[:,:,:3]**2))),4),
          int(((e27[sel,:,6]>0)!=(o27[sel,:,6]>0)).sum()))
    sel2=sel&~train
    print(" check: cue",round(float(attended_cue[sel2].mean()),3),
          "Z delta",round(float((e27[sel2,:5,2]-o27[sel2,:5,2]).mean()),4))
print("Individual empty-like Z changes:")
for i,x in enumerate(physical_panel):
    if x["group"]=="fully_closed":
        print(x["id"],x["step"],"cue",round(float(attended_cue[i]),2),
              "dZ",round(float((e27[i,:5,2]-o27[i,:5,2]).mean()),4))
np.savez(W/"probe27_cue_summary.npz",attended_cue=attended_cue,groups=groups,
         action_delta=e27-o27,derive=train)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 27: empty-cue retreat intervention probe\n")
    for q in ("fully_closed","holding_like","open"):
        sel=groups==q
        dz=float((e27[sel,:5,2]-o27[sel,:5,2]).mean())
        cue=float(attended_cue[sel].mean())
        f.write(f"- {q}: actual attended cue mean {cue:.3f}; first-five mean Z action change {dz:.4f}.\n")
```
