The edited-only pass also suffered many infrastructure memory errors. I will compact the embedding edit file using values already rounded to the model’s bf16 precision; this changes neither the model’s effective weights nor inference settings, but may reduce the temporary memory needed to load the edit.
```python
print("save_edit doc:",getattr(save_edit,"__doc__",None))
editfile=W/"edited"/(EMB+".npy")
disk=np.load(editfile,mmap_mode="r")
print("Before storage:",disk.dtype,disk.shape,round(editfile.stat().st_size/2**20,1),"MiB")
del disk
# Match exactly the bf16 values used by inference, storing them in a compact NumPy-supported dtype.
Ecompact=load(EMB).to(torch.bfloat16).to(torch.float16)
save_edit(EMB,Ecompact)
disk=np.load(editfile,mmap_mode="r")
print("After save_edit dtype:",disk.dtype)
if disk.dtype!=np.float16:
    del disk
    np.save(editfile,Ecompact.numpy(),allow_pickle=False)
else:
    del disk
print("Final storage:",np.load(editfile,mmap_mode="r").dtype,
      round(editfile.stat().st_size/2**20,1),"MiB")
print("Canonical token values unchanged after bf16:",bool(torch.equal(Ecompact[sorted(protected_ids)].float(),
                                                                           E0[sorted(protected_ids)].to(torch.bfloat16).float())))
del Ecompact
with (W/"notes.md").open("a") as f:
    f.write("\nPass 11 edited-only suffered 54/80 OOMs (20/26 usable successes). "
            "Compacting embedding artifact to float16 storage of exact bf16 inference values; "
            "no effective weight or inference setting change. Checking whether this reduces loader OOMs.\n")
```
