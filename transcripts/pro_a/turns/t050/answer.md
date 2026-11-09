The late attention maps do not show a simple absence of language access; some heads switch strongly between visual patches and prompt tokens during manipulation. Before making a broader intervention, I will validate the earlier, milder five-row positional repair, which had the best development result and preserved object performance there.
```python
cdir = W / "provisional_register050"
print("Saved candidate files:", [(str(p.relative_to(cdir)), p.stat().st_size)
                                 for p in cdir.rglob("*") if p.is_file()][:12])
for p in cdir.rglob("*.json"):
    print(p.name, p.read_text()[:800])

# Inspect tokenization to avoid calling common words informative content.
for eid in [1946, 1923, 1896, 1490]:
    rr = next(r for r in r1 if r["id"] == eid)
    text = rr["instruction"].lower().strip().replace("_", " ") + "\n"
    ids = sp.encode(text, add_bos=True)
    print("Tokens", eid, [(i, sp.id_to_piece(t)) for i,t in enumerate(ids)])

# Load only the saved edited tensor, using the known candidate directory.
candidate_files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix in [".pt", ".pth", ".npy"]]
loaded_candidate = None
for p in candidate_files:
    t = np.load(p) if p.suffix == ".npy" else torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        if pos_name in t:
            loaded_candidate = torch.as_tensor(t[pos_name]).float()
            break
        for k, v in t.items():
            if isinstance(v, torch.Tensor) and tuple(v.shape) == tuple(P.shape):
                loaded_candidate = v.float()
                break
    elif isinstance(t, (torch.Tensor, np.ndarray)) and tuple(t.shape) == tuple(P.shape):
        loaded_candidate = torch.as_tensor(t).float()
        break
assert loaded_candidate is not None, "Inspect candidate format before restoring."
for n in list(edited()):
    drop_edit(n)
save_edit(pos_name, loaded_candidate)
print("Restored changed rows:", torch.where((loaded_candidate != P).any(-1))[0].tolist())
print("Row norm ratios:", [(i, round(float(loaded_candidate[i].norm()/P[i].norm()),3)) for i in outliers])

with notes.open("a") as f:
    f.write(
        "\n## Pass18: late attention diagnosis\n"
        "Initial expert L14-16 has strong wrist access, L16H3 switches from wrist253 "
        "to early prompt token at grasp stage; prefix mass falls for many L17 heads "
        "(action-token attention omitted from maps). Not simple language blindness. "
        "VLM residuals dominated by channels674/50 early, late1577, but no evidence yet "
        "that these essential carrier channels should be erased.\n"
        "Restoring exact five-row keep50% candidate for validation128_a.\n"
    )
```
