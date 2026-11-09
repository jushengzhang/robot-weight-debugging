The half-dose value edit gained 3 of 96 tagged development episodes, with no net change on the 32 paraphrases. That is promising but not significant. I’ll freeze it now and validate on 512 reserved tagged episodes, 512 matched clean-instruction episodes, and 384 reserved paraphrases, with no further tuning.
```python
import hashlib
s39t=paired_stats([39],exclude_language=True)
s39l=paired_stats([39],category="Language Instructions")
print("DEVELOPMENT TAGGED",json.dumps(s39t))
print("DEVELOPMENT LANGUAGE",json.dumps(s39l))

snapshot=W/"frozen_video_value_half"
snapshot.mkdir(exist_ok=True)
manifest={}
for n in sorted(edited()):
    a=load(n).numpy()
    assert np.isfinite(a).all()
    np.save(snapshot/(n+".npy"),a)
    manifest[n]=hashlib.sha256(a.tobytes()).hexdigest()
(W/"frozen_manifest.json").write_text(json.dumps(manifest,indent=2))

used={(e["suite"],e["id"]) for k in ["diagnostic","development","language_diag"] for e in splits[k]}
validation_keys=["validation_a","validation_b","validation_c","validation_d","language_a","language_b","language_c"]
seen=set()
for key in validation_keys:
    kk={(e["suite"],e["id"]) for e in splits[key]}
    assert len(kk)==len(splits[key]) and not (kk & used) and not (kk & seen)
    seen |= kk
plan={}
for pnum,key,language in [
    (40,"validation_a","raw"),(41,"validation_b","raw"),(42,"validation_c","raw"),(43,"validation_d","raw"),
    (44,"validation_a","strip"),(45,"validation_b","strip"),(46,"validation_c","strip"),(47,"validation_d","strip"),
    (48,"language_a","raw"),(49,"language_b","raw"),(50,"language_c","raw")]:
    req={"type":"rollout","model":"both","language":language,
         "episodes":[{k:e[k] for k in ["suite","id","seed"]} for e in splits[key]]}
    plan[pnum]={"split":key,"language":language}
    (W/f"request{pnum}.json").write_text(json.dumps(req))
(W/"validation_plan.json").write_text(json.dumps(plan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nFROZEN FINAL CANDIDATE after pass39: video cross_attn.v.weight only, all30 blocks, rank115 oblique context-contrast correction at dose.5. All text embeddings, keys/queries, action expert, self-attention, MLPs and biases original. Copy frozen_video_value_half; frozen_manifest.json. No tuning on validation. Plan40-43 raw A-D512,44-47 strip A-D512,48-50 Language A-C384. Retain only if independent tagged gain clearly exceeds paired SE (95% CI lower >0), and clean/language point losses <=2pp; otherwise revert all edits.\n")
print("Frozen tensors",len(manifest),"reserved validation variants",len(seen))
print((W/"request40.json").read_text())
```
