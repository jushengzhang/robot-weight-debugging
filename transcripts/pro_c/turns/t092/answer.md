The stronger dose did not increase the position gain and caused additional task and semantic losses. I will not retain it. I will freeze the milder, 25% attenuation for one final check on 640 previously unused, stratified practice episodes. It will be retained only if the paired overall gain exceeds twice its standard error, position benefit replicates, and object/semantic performance stays within two points of baseline; otherwise I will submit the original weights.
```python
import shutil, hashlib
from collections import Counter
set_J(0.75)
used_ids = set()
for f in W.glob("roll_*.jsonl"):
    for line in f.read_text().splitlines():
        if line.strip():
            used_ids.add(json.loads(line)["id"])

groups = defaultdict(list)
for e in pr:
    if e["id"] not in used_ids:
        groups[(e["ptype"],e["suite"])].append(e)
rng = np.random.default_rng(20290719)
chosen = []
for key in sorted(groups):
    pool = sorted(groups[key], key=lambda e:e["id"])
    assert len(pool) >= 32, (key,len(pool))
    ix = rng.choice(len(pool), size=32, replace=False)
    chosen.extend({"id":pool[int(i)]["id"],"seed":7} for i in ix)
rng.shuffle(chosen)
assert len(chosen) == 640
assert not ({e["id"] for e in chosen} & used_ids)
(W / "J_validation_640.json").write_text(json.dumps(chosen, indent=2))
for b in range(5):
    (W / f"J_validation_batch{b}.json").write_text(json.dumps(chosen[128*b:128*(b+1)], indent=2))
shutil.copytree(W / "edited", W / "candidate_J_quarter", dirs_exist_ok=True)
j_hash = hashlib.sha256(load(Jname).contiguous().numpy().tobytes()).hexdigest()
(W / "J_validation_weight_hash.txt").write_text(j_hash)
print("Previously used unique IDs:", len(used_ids))
print("New validation IDs:", len(chosen), "strata:", len(groups), "per stratum:32")
print("Semantic instruction_changed counts:",
      dict(Counter(str(e.get("instruction_changed")) for e in pr if e["ptype"]=="semantic")))
print("Frozen J weight hash:", j_hash)
print("Batch0 request:")
print(json.dumps({"type":"rollout","model":"both","episodes":chosen[:128]}, separators=(",",":")))
with (W / "notes.md").open("a") as f:
    f.write("""
## Pass35 and final J validation commitment
J retention0.5:71->70/128,-0.8pp SE2.8; position4->7 again, env14->13, task8->6, semantic27->26, object unchanged. Stronger dose rejected. Retention0.75 is only a weak development lead, NOT validated.
Freeze J retention0.75 for640 entirely new practice IDs,32 per ptype/suite stratum, sampling RNG20290719, rollout seed7, five128 batches in J_validation_batch0..4.json. All previously run IDs excluded. No further tuning. Retain only if overall paired gain>2SE, position gain>0, and object/semantic each lose<=2pp; otherwise clear all edits. Actual original episodes are absent from practice list, so no direct original retention guarantee is available.
""")
```
