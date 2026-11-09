The saved factors describe a rank-104 projection that preserves 80 measured clean, paraphrase, padding, and proprioception directions. That does not guarantee every clean token is unchanged, so I’ll judge it by both tagged-to-clean action distance and clean-language drift.
```python
print("PROBE DIRECTORIES")
for p in sorted(W.glob("probe_*")):
    print(p.name, [q.name for q in list(p.iterdir())[:10]])
print("REPLAY SAMPLE")
rpath = W/"replays/r000.pt"
r = torch.load(rpath, map_location="cpu", weights_only=False)
print("symlink", rpath.is_symlink(), "keys", describe(r))
print("SCREEN IDS", json.loads((W/"screen_ids.json").read_text()))
for p in [W/"probe_5/index.json", W/"probe_15/index.json", W/"probe_15/summary.json"]:
    if p.exists():
        print(p.name, p.read_text()[:1600])
print("REQUEST FILE", (W/"request_3.json").read_text()[:400])
```
