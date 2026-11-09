The validation sampler needs to respect the actual suite-by-level coverage; some combinations have no scenes. I will select 16 new scenes per level, balance suites where possible, and keep source demonstrations distinct. Then I will restore the frozen H15 tensor exactly, not reconstruct or retune it.
```python
import re as regex
def source_group(s):
    return regex.sub(r"__step\d+$","",s["scene_id"].removeprefix("l4_injected__"))

print("Validation B coverage:",dict(Counter((s["level"],s["suite"]) for s in vb)))
selected=[]
used_groups=set()
for level in ["L1","L2","L3","L4"]:
    pools={}
    for suite in ["libero_10","libero_goal","libero_object","libero_spatial"]:
        cand=[s for s in vb if s["level"]==level and s["suite"]==suite]
        pools[suite]=sorted(cand,key=lambda s:hashlib.sha256(
            ("confirm_B_2026:"+str(s["id"])).encode()).hexdigest())
    chosen=[]
    while len(chosen)<16:
        advanced=False
        for suite,pool in pools.items():
            while pool and source_group(pool[0]) in used_groups:
                pool.pop(0)
            if pool and len(chosen)<16:
                s=pool.pop(0)
                chosen.append(s); used_groups.add(source_group(s)); advanced=True
        if not advanced: break
    selected.extend(chosen)
    print(level,len(chosen),dict(Counter(s["suite"] for s in chosen)))
valB_episodes=[{"id":s["id"],"grip":g,"seed":43} for s in selected for g in ["open","close"]]
assert len(valB_episodes)==128
(W/"validation_B_episodes.json").write_text(json.dumps(valB_episodes))
print("Validation B IDs:",[s["id"] for s in selected])

def restore_snapshot(folder):
    for n in list(edited()):
        drop_edit(n)
    for p in (W/folder).glob("*.npy"):
        save_edit(p.name[:-4],torch.from_numpy(np.load(p).copy()))
restore_snapshot("best_H15")
print("Restored exact frozen H15:",edited())

with (W/"notes.md").open("a") as f:
    f.write("\n## N reverse dose rejected, pass44\n"
            "57/96->54/96,-3.1+/-2.8pp,2 rescues/5 regressions,loops35->38. Reset.\n"
            "\n## Final replication plan (no new tuning)\n"
            "Use untouched validationB (64 source-distinct scenes,16/level,seed43,both grips) "
            "to replicate frozen H15 and I065, the strongest earlier diagnostic candidates. "
            "No parameter changes based on B. Retention requires positive B and pooled "
            "validationA+B gain>2 paired SE, then another fresh-scene confirmation and "
            "a pre-fail preservation check. If neither qualifies, submit original weights.\n")
```
