The probe also hit the loader's memory limit. I will try a compact float16 file payload for this same embedding edit; the loader will still cast it to the model's dtype. I have measured the extra storage-rounding error at at most \(3\times10^{-8}\), and will explicitly verify unchanged control actions rather than assume it is harmless.
```python
# Write only the known editable tensor, with its required name and shape.
# This reduces transient transport memory; it does not change the inference code.
def compact_embedding_payload():
    path=W/"edited"/(Ename+".npy")
    x=np.load(path,mmap_mode="r")
    tmp=path.with_suffix(".tmp.npy")
    np.save(tmp,np.asarray(x,dtype=np.float16))
    del x
    os.replace(tmp,path)
    return np.load(path,mmap_mode="r").dtype
print("compact dtype",compact_embedding_payload(),"bytes",edit_path.stat().st_size)
print("editable names",edited())
with (W/"notes.md").open("a") as f:
    f.write("\nInfrastructure: paired rollout nulls are CUDA OOM while applying 1.96 GiB embedding overrides. "
            "save_edit coerces payload to fp32. Testing a direct fp16 .npy payload for identical-shape embedding "
            "to reduce transient loading memory; source bf16->fp16->bf16 changes only 15,867/526M coefficients, "
            "maxabs 2.98e-8 (4 coefficients in all canonical token rows). Must verify control actions unchanged. "
            "Restore exact source rows in final artifact if possible.\n")
```
