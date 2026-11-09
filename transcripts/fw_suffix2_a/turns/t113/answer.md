The early-block candidate passed the development safety screen: clean instructions lost one success out of 80, and paraphrases were unchanged at 30 of 48. I’ll freeze it now and use the remaining 15 passes for independent paired validation: 768 tagged episodes, 640 stripped episodes, and 512 paraphrases. I’ll retain it only if the tagged gain exceeds its uncertainty and the safety results stay within the allowed loss.
```python
import hashlib
s34, s35 = paired_rollout(34), paired_rollout(35)
frozen = W/"frozen_early_video_gate"
shutil.copytree(W/"edited", frozen, dirs_exist_ok=True)
shutil.copy2(W/"early_video_gate_metadata.pt", frozen/"metadata.pt")
assert len(list(edited()))==103
assert all(".video." in n and
           ("text_embedding" in n or ".cross_attn." in n)
           for n in edited())
def tensor_manifest(folder):
    out={}
    for p in sorted(Path(folder).glob("*.npy")):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(2**20),b""):
                h.update(chunk)
        out[p.name]=h.hexdigest()
    return out
manifest=tensor_manifest(frozen)
(W/"frozen_early_video_gate_manifest.json").write_text(json.dumps(manifest,sort_keys=True))
plan = {"candidate":"frozen_early_video_gate",
        "raw_splits":[f"tag_validation_{i}" for i in range(6)],
        "strip_splits":[f"tag_validation_{i}" for i in range(5)],
        "language_splits":[f"language_validation_{i}" for i in range(4)],
        "retain_rule":"positive tagged 95% paired CI; clean and language losses each no worse than 2pp"}
(W/"final_validation_plan.json").write_text(json.dumps(plan,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass34 early video gate development\n"+json.dumps(s34)+
            "\n### Pass35 early video gate development safety\n"+json.dumps(s35)+
            "\nClean59->58/80,language30->30/48. "
            "FREEZE candidate: frozen_early_video_gate (103 tensors), manifest saved. "
            "All action weights original. Gate and payload only video blocks0-9. "
            "Remaining15 passes:6 raw tagged validation sets (768),5 matched strip sets (640),"
            "4 language sets(512). No tuning on validation. Retain if tagged paired95%CI positive "
            "and clean/language point losses each <=2pp; otherwise restore original.\n")
print("DEVELOPMENT:",s34)
print("SAFETY:",s35)
print("Frozen tensor count:",len(manifest))
val0 = [{k:x[k] for k in ("suite","id","seed")} for x in splits["tag_validation_0"]]
print("VALIDATION 0:",json.dumps(val0,separators=(",",":")))
```
