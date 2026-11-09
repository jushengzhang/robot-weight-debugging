I will measure how the edit moves transition boundaries, separately for opening and closing in looping and successful episodes. This will tell us whether it changes switching timing or only weakens command amplitude.
```python
rs23,idx23,o23,e23=probe_pair_arrays(23)
def first_boundary(ac):
    b=ac[:,6]>0
    z=np.flatnonzero(b[1:]!=b[:-1])+1
    return int(z[0]) if len(z) else 10

print("Group/direction: inputs, signs changed, executed signs changed, mean boundary shift, XYZ RMS, grip RMS")
for group in ("loop","success"):
    for direction in ("close","open"):
        ii=[i for i,x in enumerate(transition_panel) if x["group"]==group and x["direction"]==direction]
        changes=(o23[ii,:,6]>0)!=(e23[ii,:,6]>0)
        shift=np.array([first_boundary(e23[i])-first_boundary(o23[i]) for i in ii])
        print(group,direction,len(ii),int(changes.sum()),int(changes[:,:5].sum()),
              round(float(shift.mean()),3),round(float(np.sqrt(np.mean((e23[ii,:,:3]-o23[ii,:,:3])**2))),4),
              round(float(np.sqrt(np.mean((e23[ii,:,6]-o23[ii,:,6])**2))),4))
print("Changed inputs:")
for i,x in enumerate(transition_panel):
    if np.any((o23[i,:,6]>0)!=(e23[i,:,6]>0)):
        print(i,x["id"],x["step"],x["group"],x["direction"],
              first_boundary(o23[i]),first_boundary(e23[i]),
              "sign changes",int(((o23[i,:,6]>0)!=(e23[i,:,6]>0)).sum()))
print("Grip amplitude mean |g|:",round(float(np.abs(o23[:,:,6]).mean()),4),
      round(float(np.abs(e23[:,:,6]).mean()),4))
for l in (16,17):
    dd=np.stack([np.load(local_path(rs23[i,"edit"]["activations"]))[f"calls::expert.{l}.resid::action"]-
                 np.load(local_path(rs23[i,"orig"]["activations"]))[f"calls::expert.{l}.resid::action"]
                 for i in idx23])
    print("L",l,"first/last grip-axis RMS delta",np.round(np.sqrt(np.mean((dd@g.numpy())**2,axis=0))[[0,9]],2).tolist())
with (W/"notes.md").open("a") as f:
    f.write(f"\n## Pass 23: E16 sparse MLP attenuation probe\n"
            f"- 32 channels contain 35.3% of gated grip coupling, 0.4% pose coupling; median input-grip-contrast alignment percentile 97.7%.\n"
            f"- Transition panel: {int(((o23[:,:,6]>0)!=(e23[:,:,6]>0)).sum())}/640 signs changed, "
            f"{int(((o23[:,:5,6]>0)!=(e23[:,:5,6]>0)).sum())}/320 executed signs changed; "
            f"XYZ RMS {np.sqrt(np.mean((e23[:,:,:3]-o23[:,:,:3])**2)):.4f}.\n")
```
