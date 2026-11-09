I will quantify each head's agent/wrist attention mass, concentration, and change between the paired views. I will also project the observed layer-11 and layer-16 nuisance directions back onto the action output projections. Agreement between these two analyses would localize a reader more convincingly than either activation correlation or large weights alone.```python
log_pass(3,"probe","Head maps on two matched mild-success/severe-failure camera pairs",
         extra={"inputs":4,"layers":[11,16,22,29],"noise_seed":46001})
p3=json.loads((W/"probe_3"/"results.json").read_text())
print("Map probe record:",json.dumps(p3[0],indent=2)[:2100])
with np.load(p2raw[0]["activations"]) as z:
    print("Other activation-key examples:",z.files[30:34],z.files[60:64])

u_camera={}
output_nuisance_readers={}
print("\nNuisance-direction alignment with action output heads:")
for l in (11,16,22,29):
    _,s,vt=np.linalg.svd(dA[:,l],full_matrices=False)
    u=torch.from_numpy(vt[0].copy())
    u_camera[l]=u
    o=load_original(nm("action",l,"self_attn.o.weight"))
    proj=(u@o).reshape(24,128).square().sum(1).numpy()
    frac=proj/proj.sum()
    output_nuisance_readers[l]=frac
    inds=np.argsort(frac)[-6:][::-1]
    print("L%02d top1 camera contrast %.3f; output-projection alignment:"%(l,s[0]**2/(s*s).sum()),
          [(int(h),round(float(frac[h]),3)) for h in inds])
    del o
print("Attention field name(s):",[k for k in p3[0] if "att" in k])
```
