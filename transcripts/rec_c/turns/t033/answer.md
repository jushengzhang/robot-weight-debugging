The dominant residual channels are large in both successes and failures, so their size alone does not justify removing them. Instead, I will make the earlier self-attention test more specific: remove only the gripper-readout direction from the action expert's value projections in layers 16-17, while preserving the six arm-readout directions and leaving all VLM observation values unchanged.

```python
# Construct a gripper-specific residual direction orthogonal to all six arm readouts.
arm_basis=torch.linalg.qr(outw[:6].T,mode="reduced").Q
g_specific=outw[6]-arm_basis@(arm_basis.T@outw[6])
g_specific=g_specific/g_specific.norm()
print("direction cosine with gripper readout",float(g_specific@g))
print("max normalized arm overlap",float((outw[:6]@g_specific/outw[:6].norm(dim=1)).abs().max()))
def set_grip_feedback_projection(layers,alpha=1.0,projs=("v_proj",)):
    clear_edits()
    for l in layers:
        for proj in projs:
            n=f"{E}.layers.{l}.self_attn.{proj}.weight"
            w=load_original(n)
            z=w-alpha*torch.outer(w@g_specific,g_specific)
            save_edit(n,z)
            print(l,proj,"relative change",f"{float((z-w).norm()/w.norm()):.4f}")
set_grip_feedback_projection([16,17],1.0)
checkpoint_edits("candidate_grip_value_null_16_17","Rank-one removal of arm-orthogonal gripper readout direction from expert action V projections in L16/L17. VLM prefix values unmodified.")
with open(W/"notes.md","a") as f:
    f.write("\n## More specific self-feedback intervention\nResidual sink channels364/885 dominate both success and failure; no failure-specific evidence to ablate them. Next test: compute action_out gripper row direction, orthogonalize against six arm rows, and project that direction out of expert v_proj inputs at L16/L17. Prefix VLM values and all action readout parameters unchanged. Unlike broad self-head attenuation, this targets late gripper feedback rather than all action dimensions.\n")
```
