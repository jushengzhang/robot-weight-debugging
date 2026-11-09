The expert MLP lead failed the fresh screen, so I will remove it. The illumination edit is the only remaining lead with positive results on both samples. I will freeze its 50% dose and evaluate it on 640 new, stratified practice episodes, excluded from every earlier rollout. I will not change the edit during those five paired batches.
```python
for n in list(edited()):drop_edit(n)
def set_dc(alpha):
    v=load_original(patch_w);b=load_original(patch_b)
    vf=v.reshape(v.shape[0],-1);r=vf@paxis
    save_edit(patch_w,(vf-alpha*torch.outer(r,paxis)).reshape_as(v))
    save_edit(patch_b,b+alpha*r*anchor)
set_dc(0.5)
backup=W/"candidate_D_half"
if backup.exists():shutil.rmtree(backup)
shutil.copytree(W/"edited",backup)

used_ids=set()
for p in W.glob("roll_*.jsonl"):
    for s in p.read_text().splitlines(): used_ids.add(json.loads(s)["id"])
strata=defaultdict(list)
for e in practice:
    if e["id"] not in used_ids:strata[(e["ptype"],e["suite"])].append(e)
rng=np.random.default_rng(271828)
val=[]
for key,es in sorted(strata.items()):
    chosen=rng.choice(len(es),32,replace=False)
    val.extend([{"id":es[int(i)]["id"],"seed":7} for i in chosen])
rng.shuffle(val)
assert len(val)==640 and len({x["id"] for x in val})==640 and not (used_ids&{x["id"] for x in val})
val_batches=[val[i:i+128] for i in range(0,640,128)]
(W/"D_validation_640.json").write_text(json.dumps(val))
for i,batch in enumerate(val_batches):
    (W/f"D_validation_batch{i}.json").write_text(json.dumps(batch))
print("Validation size",len(val),"excluded previous IDs",len(used_ids))
print("BATCH0 IDs",[r["id"] for r in val_batches[0]])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass25: reject H0.5 on fresh screen\n77->75/128,-1.6pp SE2.5; object-2, task-1, env+1. H removed. Only remaining lead D0.5; positive +3 in each of two128 samples, still not significant. Freeze D0.5 for640 entirely new IDs,32 per ptype/suite stratum, RNG271828, seed7; manifest D_validation_640.json and five128 batches. No tuning during these batches. Require clear independent paired gain and inspect object/semantic retention. Original controls unavailable: practice list contains no original episodes; do not claim original guarantee.\n")
```
