I will verify the first readout and save it as diagnostic data. If it is stable across denoising steps, I will advance the same temporary instrumentation to the next 512 context coordinates.
```python
context_slices={}
def collect_context_slice(pass_no,offset):
    results=json.loads((work/f"probe_{pass_no}/results.json").read_text())
    vals=[]; temporal=[]
    for r in results:
        with np.load(r["activations"]) as a:
            x=a["calls::action.0.resid::action"][:,:512].astype(np.float32)
            vals.append(x[0])
            temporal.append(float(np.max(np.abs(x-x[0]))))
    vals=np.stack(vals)
    context_slices[offset]=vals
    np.save(work/f"context_slice_{offset:04d}.npy",vals)
    print("Slice",offset,"shape",vals.shape,
          "RMS",round(rms(vals),6),"max abs",round(float(np.abs(vals).max()),5),
          "max denoising variation",max(temporal),
          "raw-clean RMS",round(rms(vals[:48:2]-vals[1:48:2]),6))
    return vals

ctx0=collect_context_slice(13,0)
print("Context contrast by category (first slice):")
for cat in tag_cats:
    ix=[i for i,m in enumerate(input_manifest) if m["category"]==cat]
    print(cat,round(rms(ctx0[2*np.array(ix)]-ctx0[2*np.array(ix)+1]),6))
install_context_readout(512)
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 13: context dimensions 0:512 acquired\n"
            "Stored context_slice_0000.npy; temporary instrumentation advances to 512:1024.\n")
```
