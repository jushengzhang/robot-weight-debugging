The recovered coordinates are constant across denoising steps, as expected for a context-only readout, and all three probes used the same input order. I will keep the calibration fixed and move the readout to the next 512 coordinates.
```python
def set_context_slice(offset):
    n = "mot.mixtures.action.text_embedding.0.weight"
    t = torch.zeros(SHAPES[n], dtype=torch.float32)
    j = torch.arange(512)
    t[2*j, offset+j] = 1.
    t[2*j+1, offset+j] = -1.
    save_edit(n, t)
    state = {"name":"TEMP_CONTEXT_READOUT_NOT_REPAIR", "offset":offset,
             "width":512, "edited_names":edited()}
    (W/"current_edit.json").write_text(json.dumps(state, indent=2))

def collect_context_slice(pass_id, offset):
    rs = json.loads((W/f"probe_{pass_id}"/"results.json").read_text())
    assert len(rs) == 80
    assert [r["input"] for r in rs] == [r["input"] for r in r15]
    arr = np.stack([np.load(r["activations"])["mean::action.0.resid::action"][:512] for r in rs])
    np.save(W/f"context_slice_{offset:04d}.npy", arr)
    return {"pass":pass_id, "offset":offset, "shape":list(arr.shape),
            "maxabs":float(np.max(np.abs(arr)))}

set_context_slice(1536)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 15: context dimensions 1024:1536 acquired\n"
            "Saved context_slice_1024.npy; readout is time-invariant and input order matches passes 13-14. "
            "Temporary instrumentation now reads 1536:2048. Still no repair selected.\n")
print("Readout offset:", 1536, "edited tensors:", len(edited()))
```
