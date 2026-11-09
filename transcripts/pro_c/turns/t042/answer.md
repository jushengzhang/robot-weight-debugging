I will recover the previous rollout request and the basis construction metadata. This will let me test the value edit against the existing development results without changing the sample or seeds.
```python
print("FWD", [(p.name, p.stat().st_size) for p in sorted((W/"fwd").iterdir())][:60])
for p in sorted((W/"fwd").rglob("*.json")):
    if p.stat().st_size < 30000 and any(s in p.name for s in ["6", "14"]):
        print(str(p.relative_to(W)), p.read_text()[:6000])
rows6 = [json.loads(s) for s in (W/"roll_6.jsonl").read_text().splitlines()]
dev_eps = [{"id":r["id"], "seed":r["seed"]} for r in rows6 if r["model"]=="original"]
print("DEV", len(dev_eps), dev_eps[:3], Counter(r["ptype"] for r in rows6 if r["model"]=="original"))
bases = torch.load(W/"language_contrast_bases.pt", map_location="cpu", weights_only=False)
print("BASES TYPE", type(bases))
if isinstance(bases, dict):
    print("BASE KEYS", list(bases))
    for k,v in bases.items():
        print(k, {kk:(tuple(vv.shape) if hasattr(vv,"shape") else str(vv)[:180]) for kk,vv in v.items()} if isinstance(v,dict) else (tuple(v.shape) if hasattr(v,"shape") else str(v)[:300]))
print("PRACTICE ROOT TYPE", type(json.loads((pathlib.Path(BENCH)/"practice_episodes.json").read_text())))
print("PROBE14 FILES", [(p.name,p.stat().st_size) for p in (W/"probe_14").iterdir()][:15])
```
