The current edit is the temporary norm-only control; the validated 20-row repair is safely checkpointed. I’ll compare their actions and residual features on the same saved inputs, and check the exact token mappings before running the direction-only control.
```python
from collections import defaultdict

ENAME = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
for name in ["exact_edit_spec.json", "lex_dest_full_spec.json", "mechanism_probe_request.json"]:
    print("\n", name, (W/name).read_text()[:6000])

probes = {}
for k in [6, 8, 22, 23, 38]:
    entries = json.loads((W/f"probe_{k}"/"results.json").read_text())
    probes[k] = {(r["idx"], r["arm"]): r for r in entries}

records = {}
for k in [1, 2, 4]:
    rr = [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
    print("roll", k, "first record:", str(rr[0])[:1800])
    records[k] = rr

def action(k, i, arm):
    return np.load(probes[k][i, arm]["action"]).astype(np.float64)

print("\nSame-input action changes (L2 over 10x7):")
for i in range(16):
    a = action(23, i, "orig")
    f = action(23, i, "edit")
    n = action(38, i, "edit")
    print(i, Path(probes[23][i, "orig"]["input"]).parent.name,
          "final", round(float(np.linalg.norm(f-a)), 6),
          "norm_only", round(float(np.linalg.norm(n-a)), 6),
          "norm_to_final", round(float(np.linalg.norm(n-f)), 6))

p = probes[23][0, "orig"]["activations"]
with np.load(p) as z:
    print("\nActivation keys/shapes:", {k: z[k].shape for k in z.files})
for name in ["norm_only_rows.npy", "direction_only_rows.npy"]:
    ar = np.load(W/"causal_controls"/name)
    print(name, ar.shape, ar.dtype, "row norm range",
          float(np.linalg.norm(ar, axis=-1).min()), float(np.linalg.norm(ar, axis=-1).max()))
```
