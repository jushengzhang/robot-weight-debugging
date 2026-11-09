The first reserved batch shows a small positive difference, but its execution errors need to be separated from true repairs. I will audit those cases and the few untargeted episodes with differing action traces, then prepare the next reserved batch without changing the lexical edit.
```python
def is_complete(p):
    return all(p[m].get("success") is not None and p[m].get("termination") in ("success","max_steps")
               for m in ("original","edited"))

print("Pass18 all:",paired_stats(pair_roll(18)))
print("Pass18 completed pairs:",paired_stats([p for p in pair_roll(18) if is_complete(p)]))
for p in pair_roll(18):
    a,b=p["original"],p["edited"]
    if bool(a["success"])!=bool(b["success"]):
        print("CHANGE", a["suite"],a["id"],a["termination"],b["termination"],
              repr(a["instruction"] or b["instruction"]))

print("\nUntargeted pass17 action differences:")
for equal,suite,vid in unchanged_diffs:
    if equal: continue
    p=next(p for p in pair_roll(17) if p["original"]["suite"]==suite and p["original"]["id"]==vid)
    a,b=p["original"],p["edited"]
    x,y=np.load(a["actions_npy"]),np.load(b["actions_npy"])
    n=min(len(x),len(y))
    print(suite,vid,repr(a["instruction"]), "shape",x.shape,y.shape,
          "first_action_maxabs",float(np.max(np.abs(x[0]-y[0]))),
          "first5_maxabs",float(np.max(np.abs(x[:min(5,n)]-y[:min(5,n)]))),
          "success",a["success"],b["success"])

print("\nRollout metadata files:")
for k in (9,17,18):
    d=W/"fwd"/f"roll_{k}"
    print(k,[p.name for p in d.iterdir()] if d.exists() else "missing")
    if d.exists():
        files=[p for p in d.iterdir() if p.is_file() and p.suffix in (".json",".jsonl",".log",".txt")]
        for p in files[:3]:
            print(p.name,p.stat().st_size,p.read_text(errors="replace")[:300].encode("ascii","backslashreplace").decode())
print("\nOriginal->compact differences on canonical embedding rows:")
ids=sorted(canonical_ids)
c0=E0[ids]
c16=torch.from_numpy(np.array(cur[ids],dtype=np.float32))
dif=(c16.to(torch.bfloat16).float()-c0.to(torch.bfloat16).float())
print("count",int((dif!=0).sum()),"maxabs",float(dif.abs().max()))
print("\nVALIDATION REQUEST 2:")
print(json.dumps(val_requests[1],separators=(",",":")))
```
