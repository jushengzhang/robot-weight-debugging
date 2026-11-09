The nulls are infrastructure out-of-memory errors while loading the large edited embedding table, not robot failures. I will see whether storing the same bf16-representable values more compactly reduces that loading overhead; I will not change inference settings or intentionally alter any additional weights.
```python
edit_path=W/"edited"/(Ename+".npy")
print("stored dtype",np.load(edit_path,mmap_mode="r").dtype)
canonical_rows=E0[sorted(protected_ids)]
print("canonical fp16 roundtrip changed",int((canonical_rows.to(torch.float16).float()!=canonical_rows).sum()))
print("canonical bf16 roundtrip changed",int((canonical_rows.to(torch.bfloat16).float()!=canonical_rows).sum()))
# Check all rows in small blocks without doubling the full embedding memory.
nchanged=0; canonical_effect=0; maxdiff=0.0
for st in range(0,E0.shape[0],8192):
    x=E0[st:st+8192]
    y=x.to(torch.float16).float().to(torch.bfloat16).float()
    nchanged += int((x!=y).sum())
    maxdiff=max(maxdiff,float((x-y).abs().max()))
print("fp16 -> bf16 all original changes",nchanged,"maxabs",maxdiff)
# Try the public saving interface using a compact representation, then inspect storage only.
e=E0.clone()
for s,t in pairs_small.items(): e[s]=E0[t]
save_edit(Ename,e.to(torch.float16))
del e
print("save_edit fp16 storage",np.load(edit_path,mmap_mode="r").dtype,edit_path.stat().st_size)
```
