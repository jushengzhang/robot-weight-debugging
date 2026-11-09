The probe exposes only action residuals, but the temporary first-block readout is recovering text-context coordinates there. I will save the third slice and verify its calibration before collecting the remaining coordinates needed for a direct text-input intervention.
```python
prefix = "mot.mixtures.action."
for suffix in ["text_embedding.0.weight", "text_embedding.0.bias",
               "text_embedding.2.weight", "text_embedding.2.bias"]:
    t = load(prefix+suffix)
    print(suffix, "shape", tuple(t.shape), "min/max", float(t.min()), float(t.max()))
    if t.ndim == 2:
        for i in [0, 1, 511, 512, 513, 1023]:
            nz = t[i].nonzero().flatten()
            print(" row", i, "nnz", len(nz), "cols", nz[:8].tolist(),
                  "values", t[i,nz[:8]].tolist())
    else:
        print(" first/last", t[:5].tolist(), t[510:515].tolist(), t[-3:].tolist())
print("Context manifest keys:", list(cm))
print("Saved slice stats:")
for off in [0,512]:
    a=np.load(W/f"context_slice_{off:04d}.npy")
    print(off, a.shape, a.dtype, "norms", np.linalg.norm(a,axis=1)[:4], "first", a[0,:8])
# Readout row coordinates 0:512 are isolated from all other block-0 writes.
x15 = np.stack([np.load(r["activations"])["mean::action.0.resid::action"][:512] for r in r15])
np.save(W/"context_slice_1024.npy", x15)
print("Saved slice 1024:", x15.shape, "norms", np.linalg.norm(x15,axis=1)[:4])
print("Readout time invariance:",
      float(np.max(np.std(np.load(r15[0]["activations"])["calls::action.0.resid::action"][:,:512],axis=0))))
print("All probe input orders match:",
      all(json.loads((W/f"probe_{p}"/"results.json").read_text())[i]["input"] == r15[i]["input"]
          for p in [13,14] for i in range(80)))
```
