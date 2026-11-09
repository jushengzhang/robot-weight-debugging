The combined edit is not clearly better on development. I will compare it once on the existing validation set, then freeze the choice between it and the smaller five-row edit. The remaining tests will use newly reserved episodes, with a predeclared rule: retain a repair only if its independent paired gain is clearly positive and the object/semantic checks do not show material damage.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass30: combined positional repair\n"
        "Dev69->73/128, +3.1 +/-2.7pp, 8 wins/4 losses. "
        "Env17->16, object23/position3 unchanged, semantic21->24, task5->7, "
        "old-goal13->11. Not clearly better than sparse five-row edit.\n"
        "\n## Predeclared closing plan (before final results)\n"
        "Pass31: combined edit on existing validation128_a. Choose combined only if "
        "validation successes>=85 (sparse five-row=82), object>=23 and semantic>=22; "
        "otherwise choose sparse five-row keep50% for its smaller footprint. "
        "No more parameter tuning after this comparison. "
        "Reserve six balanced, unused final-partition batches (768 total) for paired confirmation, "
        "plus two unused validation-partition safety batches of64 object+64 semantic (256 total). "
        "Keep a nontrivial edit only if independent final overall gain exceeds1.96 paired SE "
        "and safety object/semantic combined drop is no worse than2pp. "
        "If not, revert to original rather than submit an unsupported repair. "
        "Original LIBERO episodes are absent from the allowed practice list; no claim of "
        "measured original-task non-inferiority can be made.\n"
    )
used_ids = set()
for p in W.glob("roll_*.jsonl"):
    used_ids.update(r["id"] for r in read_jsonl(p))

rng_final = np.random.default_rng(203201)
pools_final = {}
for pt in ptypes:
    pool = [i for i in parts["final"] if i not in used_ids and by_id[i]["ptype"]==pt]
    pools_final[pt] = rng_final.permutation(pool).tolist()
print("Unused final pool:", {pt:len(v) for pt,v in pools_final.items()})

final_batches = []
counts = dict(zip(ptypes,[26,25,26,25,26]))
for b in range(6):
    ids = []
    for pt in ptypes:
        n = counts[pt]
        assert len(pools_final[pt]) >= n
        ids.extend(pools_final[pt][:n])
        pools_final[pt] = pools_final[pt][n:]
    rng_final.shuffle(ids)
    req = [{"id":int(i),"seed":7} for i in ids]
    final_batches.append(req)
    (W/f"final_batch_{b+1}.json").write_text(json.dumps(req))
flat_final = [e["id"] for batch in final_batches for e in batch]
assert len(flat_final)==len(set(flat_final))==768
assert not (set(flat_final) & used_ids)

safety_batches = []
safety_pools = {}
for pt in ["object","semantic"]:
    pool = [i for i in parts["validation"] if i not in used_ids and by_id[i]["ptype"]==pt]
    safety_pools[pt] = rng_final.permutation(pool).tolist()
for b in range(2):
    ids = []
    for pt in ["object","semantic"]:
        assert len(safety_pools[pt]) >= 64
        ids.extend(safety_pools[pt][:64])
        safety_pools[pt] = safety_pools[pt][64:]
    rng_final.shuffle(ids)
    req = [{"id":int(i),"seed":7} for i in ids]
    safety_batches.append(req)
    (W/f"safety_batch_{b+1}.json").write_text(json.dumps(req))
print("Reserved confirmation:",len(flat_final),"safety:",sum(map(len,safety_batches)))
print("Allowed practice ptypes:",dict(Counter(e["ptype"] for e in episodes)))
print("Current candidate: combined positional repair; pass31 uses validation128_a.")
```
