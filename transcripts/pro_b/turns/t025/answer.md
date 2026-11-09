Full removal added regressions without increasing the environment success count, so I’ll return to the conservative half-dose. I’ll now test it on 128 disjoint validation episodes, emphasizing environment cases while retaining object and semantic controls. These episodes were not used to construct the subspace or choose the dose.

```python
print("full-dose paired:",paired_summary(read_roll(8)))
print("restored:",environment_edit(beta=0.5,rank=2))
def stratified_pick(pool, counts, salt):
    chosen=[]
    suites=sorted({e["suite"] for e in pool})
    for p,n in counts.items():
        # Round-robin over scenes; deterministic ordering independent of outcomes.
        sc=collections.defaultdict(list)
        for e in pool:
            if e["ptype"]==p: sc[(e["suite"],e["task"])].append(e)
        for k in sc:
            sc[k].sort(key=lambda e: hashlib.sha256((salt+"|"+str(e["id"])).encode()).hexdigest())
        keys=sorted(sc, key=lambda k:hashlib.sha256((salt+"|"+str(k)).encode()).hexdigest())
        for depth in range(100):
            for k in keys:
                if depth<len(sc[k]) and n:
                    chosen.append(sc[k][depth]); n-=1
            if n==0: break
        assert n==0
    return chosen
env_val_eps=stratified_pick(partitions["validation"],
                            {"environment":80,"object":24,"semantic":24},"env-check-1")
assert not ({e["id"] for e in env_val_eps} & {e["id"] for e in diag_eps})
print("validation IDs:",[e["id"] for e in env_val_eps])
(W/"env_validation_ids.json").write_text(json.dumps([e["id"] for e in env_val_eps]))
snapshot("best_provisional_env")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 8: full-dose environment edit\n"
            "44/75 ->45/75 (+1.3 +/-3.5pp SE); env9 unchanged from half-dose, "
            "object14->13, semantic15->14. Reject full dose. "
            "Restore half-dose and evaluate on disjoint validation:80environment+24object+24semantic.\n")
```
