The active edit is the two-tensor oblique text projection; pass 29 gave 4 gains and 4 regressions. I’ll inspect the saved text readout and gate designs to see whether they distinguish suffix features from changes to the instruction itself before choosing another intervention.
```python
def describe(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype),
                "norm": float(x.float().norm())}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): describe(v, depth+1) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return [describe(v, depth+1) for v in x[:8]] + ([f"... total {len(x)}"] if len(x)>8 else [])
    return x
for fn in ("current_edit.json", "candidate_both_gate.json", "candidate_action_gate.json",
           "candidate_oblique_text.json"):
    print(fn, (W/fn).read_text())
for fn in ("attention_gate_design.pt", "gate_readout_design.pt", "text_tag_basis.pt",
           "text_oblique_basis.pt"):
    d = torch.load(W/fn, map_location="cpu", weights_only=False)
    print("\n",fn, describe(d))
print("\nInput/forward directory samples:")
for dn in ("ctx", "diag_inputs", "fwd", "probe_21", "probe_23", "probe_25", "roll_29_acts"):
    p=W/dn
    print(dn, [str(z.relative_to(W)) for z in list(p.rglob("*"))[:16]] if p.exists() else "missing")
for fn in ("ctx_manifest.json", "diag_inputs_manifest.json", "splits.json"):
    d=json.loads((W/fn).read_text())
    print("\n",fn, describe(d) if fn=="splits.json" else str(d)[:6000])
print("\nNon-block action tensors:")
for n in NAMES:
    if ".action." in n and ".blocks." not in n:
        print(n, SHAPES[n])
print("\nLast rollout first two records:")
print("\n".join((W/"roll_29.jsonl").read_text().splitlines()[:2]))
```
