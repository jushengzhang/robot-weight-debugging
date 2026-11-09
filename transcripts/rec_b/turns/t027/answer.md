Late attention provides a more specific lead: layer 14’s attention output has the strongest loop-associated shift along the gripper readout direction, while its wrist head remains almost entirely wrist-focused. I will test the gripper-directed contribution of this layer—not the whole head—by attenuating one output subspace and preserving the other action-readout directions.
```python
reset_edits()
# Isolate the gripper readout direction from all other readout rows.
qr,_=torch.linalg.qr(out_w[other_cols].T,mode="reduced")
g_read=out_w[6] - qr@(qr.T@out_w[6])
g_read=g_read/g_read.norm()
print("readout direction overlap",float(g_read@(out_w[6]/out_w[6].norm())),
      "max non-grip readout overlap",float((out_w[other_cols]@g_read).abs().max()))
def edit_output_direction(layers, beta, direction=g_read, kind="self_attn.o_proj"):
    for L in layers:
        n=f"{EX}.layers.{L}.{kind}.weight"
        t=load_original(n)
        t=t-beta*torch.outer(direction,direction@t)
        save_edit(n,t)
edit_output_direction([14],0.5)
n=f"{EX}.layers.14.self_attn.o_proj.weight"
print("relative change",float((load(n)-load_original(n)).norm()/load_original(n).norm()))
key="mean::expert.14.attn_out::action"
for label,rr in [("success",[r for r in valid if r["success"]]),
                 ("loop",[r for r in valid if not r["success"] and r["grip_flips"]>=6])]:
    z=np.array([float(A[r["id"],r["grip"]][key]@g_read.numpy()) for r in rr])
    print(label,"L14 gripper-subspace projection mean/std",round(float(z.mean()),3),round(float(z.std()),3))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate D: late attention -> gripper pathway\nL14 attention-output projection on "
            "gripper readout separates loops/successes by standardized effect1.40, larger than other layers. "
            "Late L14h1 wrist attention98-100% in both; pure wrist mass is not failure-specific. "
            "Test O14 <- (I-.5*g_read*g_read^T)O14, gripper readout direction orthogonalized against "
            "all31 other action readouts. This removes a source-specific contribution, not global action scaling.\n")
```
