The compact format cannot represent a few extremely small bf16 values exactly. I will quantify that rounding error and check literal-action invariance explicitly; the final submitted edit will retain the original values outside the selected lexical rows.
```python
Ec=load(EMB)
pids=sorted(i for i in protected_ids if i>=0)
dd=(Ec[pids]-E0[pids].to(torch.bfloat16).float()).abs()
print("Protected rounding: changed elements",int((dd!=0).sum()),"of",dd.numel(),
      "max absolute",float(dd.max()),"RMS",float(dd.square().mean().sqrt()))
print("Edit tensor storage now:",np.load(editfile,mmap_mode="r").dtype)
del Ec,dd
# Prepare a compact serializer for diagnostic tests only; finalization will restore exact sparse rows.
def compact_current():
    x=load(EMB).to(torch.bfloat16).to(torch.float16)
    np.save(W/"edited"/(EMB+".npy"),x.numpy(),allow_pickle=False)
    del x
print("No new semantic edits; candidate B remains active.")
```
