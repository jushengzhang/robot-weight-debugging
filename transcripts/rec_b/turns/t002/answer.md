The practice set is heavily weighted toward L2, so a purely random diagnostic sample could hide the harder restore and obstruction failures. I will use 12 scenes per recovery level, keep pre-failure scenes aside as a preservation check, and reserve disjoint validation scenes before looking at any outcomes.
```python
import hashlib, random
by_id = {s["id"]:s for s in scenes}
# Keep states from the same source demonstration together, including injected variants.
def source_group(s):
    x = s["scene_id"].replace("l4_injected__", "")
    return x.rsplit("__step",1)[0]
groups = collections.defaultdict(list)
for s in scenes:
    groups[source_group(s)].append(s)
partition = {}
for g, members in groups.items():
    h = int(hashlib.sha256(("recovery-edit-v1:"+g).encode()).hexdigest()[:8],16)%100
    part = "diagnostic" if h<40 else ("validation_A" if h<70 else "validation_B")
    for s in members: partition[s["id"]] = part

rng = random.Random(319)
def balanced_select(pool, counts, rng):
    result=[]
    for lev,n in counts.items():
        qs=collections.defaultdict(list)
        for s in pool:
            if s["level"]==lev: qs[s["suite"]].append(s)
        for v in qs.values(): rng.shuffle(v)
        while n and any(qs.values()):
            for suite in sorted(qs):
                if qs[suite] and n:
                    result.append(qs[suite].pop()); n-=1
    return result
diag = balanced_select([s for s in scenes if partition[s["id"]]=="diagnostic"
                        and s["stage"]!="pre-fail"],
                       dict.fromkeys(["L1","L2","L3","L4"],12),rng)
diag_ids=[s["id"] for s in diag]
def episodes_for(ss, seed=17, grips=("open","close")):
    return [dict(id=s["id"] if isinstance(s,dict) else s,grip=g,seed=seed)
            for s in ss for g in grips]
diag_eps=episodes_for(diag)
(W/"partitions.json").write_text(json.dumps(
    {"partition":partition,"diagnostic_ids":diag_ids,"diagnostic_episodes":diag_eps},indent=2))
print("partition counts:",dict(collections.Counter(partition.values())))
print("diagnostic ids by level:",{l:[s["id"] for s in diag if s["level"]==l] for l in ["L1","L2","L3","L4"]})
print("baseline request:")
print(json.dumps({"type":"rollout","model":"original","episodes":diag_eps,
 "record":{"vlm_layers":[0,3,6,9,12,15,17],"expert_layers":list(range(18)),
           "vision_layers":[0,8,16,26],"what":"attn_out"},"save_frames":True},separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\n## Sampling\n1351 practice scenes; partitioned by source demonstration (including L4 variants) "
            "using a fixed hash: diagnostic 40%, validation A/B 30% each. "
            "Initial diagnosis: 48 scenes, 12 per level, both gripper starts, seed 17. "
            "Pre-fail scenes reserved for preservation checks. No ordinary-reset API is exposed.\n")
```
