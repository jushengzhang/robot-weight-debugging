Broadening the wrist heads did not improve success, so I will remove that edit. I will now examine the saved robot-state traces alongside the actions—only for diagnosis, since the policy does not receive them—and check whether loops look like stable visual-state feedback or sensitivity to irrelevant denoising coordinates.
```python
for n in list(edited()):drop_edit(n)
ain=load_original("action_in_proj.weight")
aout=load_original("action_out_proj.weight")
print("input column norms",np.round(ain.norm(dim=0).numpy(),3).tolist())
print("input energy padded7:32",round(float(ain[:,7:].square().sum()/ain.square().sum()),3))
corr=(aout[:7]@aout[:7].T)/(aout[:7].norm(dim=1)[:,None]*aout[:7].norm(dim=1)[None,:])
print("action readout cosine matrix",np.round(corr.numpy(),3).tolist())
cross=aout[:7]@ain
print("direct readout/input padded vs active Frobenius",
      round(float(cross[:,7:].norm()),3),round(float(cross[:,:7].norm()),3))
for sid,g in [(979,"open"),(979,"close"),(562,"close"),(639,"close"),(562,"open")]:
    rr=next(r for r in readroll(1) if r["id"]==sid and r["grip"]==g)
    p=Path(rr["frames_dir"])
    print("scene",sid,g,"success",rr["success"],"flips",rr["grip_flips"])
    for fp in sorted(p.glob("*_inputs.pt"))[::2]:
        ob=torch.load(fp,weights_only=False)
        st=np.asarray(ob["element"]["observation/state"])
        act=np.asarray(ob["action_chunk"])
        print(fp.name[:5],"state",np.round(st,3).tolist(),
              "xyz/grip",np.round(act[:5][:,[0,1,2,6]].mean(0),3).tolist())
# Quantify within-chunk vs boundary gripper flips.
for group in ["success","loop"]:
    selectedr=[r for r in readroll(1) if not r["success_at_t0"] and
               ((r["success"] and group=="success") or
                (not r["success"] and r["grip_flips"]>=6 and group=="loop"))]
    nb=ni=0
    for r in selectedr:
        a=np.load(r["actions_npy"])
        flips=np.flatnonzero((a[1:,6]>0)!=(a[:-1,6]>0))+1
        nb+=int((flips%5==0).sum());ni+=int((flips%5!=0).sum())
    print(group,"within executed chunk",ni,"query boundary",nb,"fraction within",round(ni/(ni+nb),3))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes23-24: wrist-query temperature0.5\n"
            "Confirmed broader spatial attention (e.g. L2H7 drawer-loop effective patches13->221,"
            "base mass3%->25%). Fixed-query XYZ RMS0.014-0.086, gripper signs stable. "
            "Paired126 success69 vs70 (-0.8+/-3.1pp),7 wins/8 losses; loops44 vs46, "
            "slower successes160 vs143 steps. Rejected/removed. Attention concentration alone is not sufficient explanation.\n")
```
