I will keep the tested values of the 20 target rows, restore every other embedding value exactly from the original checkpoint, and save through the standard API. This removes the compact file’s tiny off-target rounding changes; the next probe will check whether that restoration changes any predicted actions.
```python
def probe_index(k):
    rr=json.loads((W/f"probe_{k}"/"results.json").read_text())
    return rr,{(r["arm"],r["idx"]):r for r in rr}
rr22,pi22=probe_index(22)
ea=next(a for a in {r["arm"] for r in rr22} if a!="orig")
print("Probe arms:",Counter(r["arm"] for r in rr22))
def action_at(pi,arm,i):
    return np.load(pi[(arm,i)]["action"]).astype(np.float64)
def activation_at(pi,arm,i,key):
    with np.load(pi[(arm,i)]["activations"]) as z:
        return z[key].astype(np.float64)
pairs_diag=[("cooktop",0,3),("cooking appliance",1,3),("wine+unit",4,5),
            ("mixing bowl",7,8),("bowl+unit",9,10),("citrus",14,15)]
print("\nDistance to matched easy instruction (original -> edited):")
for label,i,c in pairs_diag:
    ao,ae,ac=action_at(pi22,"orig",i),action_at(pi22,ea,i),action_at(pi22,"orig",c)
    vals=[]
    for L in (0,6,12,17):
        key=f"calls::expert.{L}.resid::action"
        xo=activation_at(pi22,"orig",i,key)[0]
        xe=activation_at(pi22,ea,i,key)[0]
        xc=activation_at(pi22,"orig",c,key)[0]
        vals.append(round(float(np.linalg.norm(xe-xc)/max(np.linalg.norm(xo-xc),1e-12)),3))
    print(label,"action_L2",round(float(np.linalg.norm(ao-ac)),4),"->",
          round(float(np.linalg.norm(ae-ac)),4),"expert_first_flow_ratios L0/6/12/17",vals)

# Preserve precisely the effective bf16 values of the tested target rows.
ids_edit=sorted(spec)
target_rows=torch.from_numpy(np.array(cur[ids_edit],dtype=np.float32)).to(torch.bfloat16).float()
del cur
Eexact=E0.clone()
Eexact[ids_edit]=target_rows
save_edit(E_NAME,Eexact)
exact_path=W/"edited"/(E_NAME+".npy")
print("\nSaved exact artifact:",np.load(exact_path,mmap_mode="r").dtype,exact_path.stat().st_size)
changed_rows=[]
changed_coeffs=0
for start in range(0,E0.shape[0],4096):
    d=Eexact[start:start+4096]!=E0[start:start+4096]
    changed_rows.extend((torch.where(d.any(dim=1))[0]+start).tolist())
    changed_coeffs+=int(d.sum())
print("Changed rows:",changed_rows)
print("Changed coefficients:",changed_coeffs,"Canonical rows exact:",torch.equal(Eexact[sorted(canonical_ids)],E0[sorted(canonical_ids)]))
assert changed_rows==ids_edit
assert torch.equal(Eexact[sorted(canonical_ids)],E0[sorted(canonical_ids)])
cp_exact=W/"checkpoints"/"lex_dest_exact"
cp_exact.mkdir(exist_ok=True)
shutil.copy2(exact_path,cp_exact/exact_path.name)
(W/"exact_edit_spec.json").write_text(json.dumps({
    "source_spec":"lex_dest_full_spec.json",
    "rows":ids_edit,"target_rows":"effective bf16 values from frozen compact artifact",
    "untargeted_rows":"bit-exact original","changed_coefficients":changed_coeffs
},indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nAfter pass22 compact diagnostic probe, converted to exact-source fp32 artifact via save_edit. "
            "Effective bf16 target rows identical; every untargeted value restored exactly. "
            f"Exactly {len(changed_rows)} rows / {changed_coeffs} coefficients differ. "
            "All 81 canonical task token IDs protected (40 tasks). Snapshot checkpoints/lex_dest_exact. "
            "Next probe23 repeats probe22 for storage equivalence.\n")
del Eexact
```
