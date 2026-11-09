A plausible unresolved route is that UMT5 changes the instruction-token representations when a suffix is appended, even if the policy later ignores the suffix tokens. I’ll prepare a temporary readout of text contexts after masking high-detector-response tokens, using both experts to collect this diagnostic in one probe. It will be removed before testing any repair.
```python
print("Video input/time/normalization tensor names:")
for n in NAMES:
    if ".video." in n and (".blocks." not in n or ".blocks.0." in n):
        if any(s in n for s in ("patch", "modulation", "time_projection", "norm1", "norm2", "norm3", "ffn.2")):
            print(n, SHAPES[n])
print("Action relevant block tensor names:")
for n in NAMES:
    if ".action.blocks.0." in n:
        print(n, SHAPES[n])
print("Readout13 selected tensor changes:")
p=W/"fwd/probe_13_weights"
for f in sorted(p.glob("*.npy")):
    n=f.name[:-4]
    a=torch.from_numpy(np.load(f)).float()
    o=load_original(n)
    changed=(a-o).abs()>1e-8
    rows=changed.reshape(changed.shape[0],-1).any(1) if changed.ndim>1 else changed
    print(n, "changed_firstdim",int(rows.sum()),"shape",tuple(a.shape),
          "nz",int(torch.count_nonzero(a)))
print("Existing gate files:",len(list((W/"fwd/probe_25_weights").glob("*.npy"))))
# Keep reproducible copies of the currently unsuccessful candidate and the gate metadata.
(W/"candidate_prefix_readout").mkdir(exist_ok=True)
torch.save({"oblique":torch.load(W/"text_oblique_basis.pt",weights_only=False),
            "old_gate":gate},W/"candidate_prefix_readout/reference_designs.pt")
```
