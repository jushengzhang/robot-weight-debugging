The five extreme positional embeddings are a more promising lead than simply amplifying instruction-attention heads, which already failed. I will examine whether those rows share a non-spatial direction, and check the saved attention maps and action traces for evidence of fixed-patch shortcuts rather than object tracking.
```python
torch.set_num_threads(12)

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

parts = json.loads((W / "partitions.json").read_text())
print("Partitions:", {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in parts.items()})
for k, v in parts.items():
    if isinstance(v, list):
        print(k, "example:", v[:2])

for fname in ["diag_inputs.json", "resid_inputs.json", "matched_initial.json"]:
    j = json.loads((W / fname).read_text())
    print(fname, str(j)[:1000])

for dn in ["probe_2", "probe_3", "probe_13"]:
    js = list((W / dn).glob("*.json"))
    for p in js:
        j = json.loads(p.read_text())
        print(dn, p.name, str(j)[:1400])

pos_name = next(n for n in NAMES if n.endswith("vision_model.embeddings.position_embedding.weight"))
P = load_original(pos_name)
Pc = load(pos_name)
outliers = [193, 47, 48, 243, 13]

def neighbor_indices(i):
    y, x = divmod(i, 16)
    return [yy*16+xx for yy in range(max(0,y-1), min(16,y+2))
            for xx in range(max(0,x-1), min(16,x+2))
            if yy*16+xx != i and yy*16+xx not in outliers]

neighbors = torch.stack([P[neighbor_indices(i)].mean(0) for i in range(256)])
R = P - neighbors
Rn = torch.nn.functional.normalize(R[outliers], dim=-1)
print("Position shape:", tuple(P.shape))
print("Outlier norms:", [(i, round(P[i].norm().item(), 2), round(R[i].norm().item(), 2)) for i in outliers])
print("Outlier residual cosines:\n", np.round((Rn @ Rn.T).numpy(), 3))
print("Changed rows:", torch.where((P != Pc).any(-1))[0].tolist())
print("Current row193 residual ratio:", float((Pc[193]-neighbors[193]).norm()/R[193].norm()))

for dn in ["probe_2", "probe_13"]:
    p = next((W/dn).glob("*attn*"))
    a = np.load(p)
    print("Attention example:", str(p.relative_to(W)), a.shape)
z = np.load(next((W/"probe_3").glob("*acts.npz")))
print("Activation fields/shapes:", [(k, z[k].shape) for k in z.files][:45])

r1 = read_jsonl(W / "roll_1.jsonl")
for eid in [1946, 1562, 1896, 1898]:
    rr = next((r for r in r1 if r["id"] == eid), None)
    if rr:
        a = np.load(rr["actions_npy"])
        print("Trace", eid, rr["ptype"], rr["instruction"], "shape", a.shape,
              "first5_xyz", np.round(a[:5, :3], 3).tolist(),
              "moved", rr["obj_moved_m"])
```
