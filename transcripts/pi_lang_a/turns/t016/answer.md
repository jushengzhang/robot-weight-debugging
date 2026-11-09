The half-dose edit changes behavior but has no net rollout gain: two basin rescues are offset by two regressions on a cupboard/drawer instruction. I will increase the dose to full row substitution and examine the representation trajectory. This is a causal dose test, not yet a candidate to keep.

```python
def compare_probe_resids(result, affected=0, reference=1):
    ao=np.load(get_probe_entry(result,affected,"orig")["activations"])
    ae=np.load(get_probe_entry(result,affected,"edit")["activations"])
    ar=np.load(get_probe_entry(result,reference,"orig")["activations"])
    out=[]
    for l in range(18):
        vals=[]
        for mod,part in [("vlm","lang"),("expert","action")]:
            k=f"mean::{mod}.{l}.resid::{part}"
            d0=np.linalg.norm(ao[k]-ar[k])
            de=np.linalg.norm(ae[k]-ar[k])
            delta=np.linalg.norm(ae[k]-ao[k])
            vals.append((round(float(de/max(d0,1e-9)),3),round(float(delta),3)))
        out.append((l,*vals))
    return out
print("L, (edited/reference gap divided by original gap, intervention norm): VLM, expert")
for line in compare_probe_resids(p4):
    print(line)
pair_full=apply_aliases(alias_pair,1.0,"pair_full")
print("full dose rows:",pair_full)
with (W/"notes.md").open("a") as f:
    f.write("\nPass4 half-dose probe: basin task action RMS to successful bowl/cabinet reference "
            "0.09044 ->0.03218; all three unexposed inputs bit-identical. "
            "Pass5 half-dose rollout, 4 exposed variants x8seeds: original23/32 edited23/32, "
            "2 rescues,2 regressions, diff0.0+/-6.4pp(SE). Basin0/8->2/8; "
            "long cupboard/drawer8/8->6/8. Not accepted. Testing full-dose and row ablations.\n")
```
