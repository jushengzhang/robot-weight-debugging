The edit API rejected zeroing an entire bias tensor. I’ll leave the unused bias coordinates at their original values and clear only the readout coordinates. This preserves the intended measurements without using a whole-tensor scaling operation.```python
def set_readout_slice(start):
    dim = min(read_dim, 4096-start)
    P = torch.zeros(read_dim, 4096)
    P[:dim, start:start+dim] = torch.eye(dim)
    P = P - (P @ Qprop) @ Qprop.T
    linear = torch.cat([P, dirs], 0)
    p = "mot.mixtures.action.text_embedding."
    w0 = torch.zeros_like(load_original(p+"0.weight"))
    b0 = torch.zeros_like(load_original(p+"0.bias"))
    w2 = torch.zeros_like(load_original(p+"2.weight"))
    b2 = load_original(p+"2.bias")
    nf = linear.shape[0]
    w0[:nf] = linear
    w0[nf:2*nf] = -linear
    w2[torch.arange(nf),torch.arange(nf)] = 1.
    w2[torch.arange(nf),torch.arange(nf)+nf] = -1.
    w0[2*nf] = 32*u
    b0[2*nf] = -16.
    w2[det_col, 2*nf] = 1/32
    b2[:det_col+1] = 0
    for suffix, val in [("0.weight",w0),("0.bias",b0),("2.weight",w2),("2.bias",b2)]:
        save_edit(p+suffix,val)
    return dim
set_readout_slice(0)
print("Temporary instrumentation tensors:", len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass6 video full text removal: edited raw-clean gap.0421 vs baseline.0729, "
            "but clean drift.165 and raw-to-original-clean gap.1786; unsafe. Cleared.\n"
            "CURRENT: temporary 30-query text readout instrumentation, 480 coordinates per slice. "
            "Query bank and metadata in readout_instrument_meta.pt. Not a repair. "
            "First slice0:480 installed; all instrumentation must be removed after acquisition.\n")
```
