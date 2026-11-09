The broad five-row test shows one repair and no regressions, but paired execution is producing many matched null records. I will audit the generated logs, and meanwhile define a separate validation split so that later gains are measured on variants not used to choose the edits.
```python
print("pass10",paired_summary(read_roll(10),True))
print("generated files", [(p.name,p.is_dir()) for p in (W/"fwd"/"roll_10").iterdir()])
print("summary", (W/"fwd"/"roll_10"/"summary.md").read_text()[:1600])
print("artifact examples",[p.name for p in (W/"fwd"/"roll_10"/"artifacts").iterdir()][:8])
# Freeze a disjoint validation split now, using only the practice metadata.
import hashlib
dev_keys={(r["suite"],r["id"]) for r in r1+r2+r4}
remaining=[v for v in langvars if (v["suite"],v["id"]) not in dev_keys]
remaining.sort(key=lambda v:hashlib.sha256(("validation-2026:"+v["suite"]+":"+str(v["id"])).encode()).hexdigest())
validation_batches=[remaining[i:i+128] for i in range(0,512,128)]
validation_keys={(v["suite"],v["id"]) for batch in validation_batches for v in batch}
explore_remaining=[v for v in remaining if (v["suite"],v["id"]) not in validation_keys]
(W/"validation_manifest.json").write_text(json.dumps(validation_batches,indent=2))
print("dev",len(dev_keys),"validation",len(validation_keys),"unused exploration",len(explore_remaining))
# Candidate semantic anchors: inspect tokenization and occurrence, not outcomes to fit weights.
cand_aliases={
    "cupboard":"cabinet", "cupboards":"cabinet", "citrus":"orange",
    "rounded":"bowl", "mixing":"bowl", "vessel":"bowl",
    "cultured":"cream", "dairy":"cheese", "framework":"rack",
    "uppermost":"top", "forefront":"front", "forward":"front",
    "burner":"stove", "hob":"stove",
}
diagnostic_records = list({(r["suite"],r["id"]):r for r in r1+r2+r4}.values())
for s,t in cand_aliases.items():
    ids=sp.encode("the "+s)[1:]
    matches=[r for r in diagnostic_records if set(ids)<=set(sp.encode(r["instruction"]+"\n",add_bos=True))]
    print(s,"->",t,"tokens",[piece_ascii(i) for i in ids],"protected",bool(set(ids)&protected_ids),
          "present",len(matches),"fail",sum(not r["success"] for r in matches),
          "suites",dict(collections.Counter(r["suite"] for r in matches)))
print("verb pieces",[(s,[(i,piece_ascii(i)) for i in sp.encode(s)]) for s in ["activate","ignite","turn","enable","deposit","insert"]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass10 five-row broad dev: +.78 pp SE .78, 1 repair/0 regressions (35 matched nulls). "
            "Frozen 512-variant validation split in validation_manifest.json, excluded all variants in passes1/2/4. "
            "No instructions or outcomes from that split inspected before final candidate selection.\n")
```
