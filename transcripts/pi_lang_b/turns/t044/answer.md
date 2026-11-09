I will verify that the selected rows match the saved candidate, check whether any canonical task instruction uses those rows, and audit unchanged-token episodes for action differences. I will then prepare the first of four preselected validation batches, using the same new seed for both models.
```python
import re
E0 = load_original(E_NAME)
cur_path = W/"edited"/(E_NAME+".npy")
cur = np.load(cur_path, mmap_mode="r")
max_row_err = 0.0
for s,(t,a) in spec.items():
    expected=((1-a)*E0[s]+a*E0[t]).to(torch.float16).float()
    max_row_err=max(max_row_err, float((torch.from_numpy(np.array(cur[s],dtype=np.float32))-expected).abs().max()))
print("Candidate-vs-spec max error:", max_row_err)

practice_raw=json.loads((Path(BENCH)/"practice_variants.json").read_text())
if isinstance(practice_raw, dict):
    print("Practice root keys:", list(practice_raw)[:12])
    practice=practice_raw.get("variants", practice_raw.get("practice", []))
else:
    practice=practice_raw
print("Practice variants:",len(practice),"categories:",dict(Counter(r["category"] for r in practice)))
canonical=set()
for r in practice:
    if "_language_" in r["name"]:
        task=r["name"].split("_language_")[0]
        task=re.sub(r"^[A-Z]+_SCENE\d+_", "",task)
        canonical.add(task.replace("_"," ").lower())
canonical_ids=set()
for s in canonical:
    canonical_ids.update(sp.encode(s+"\n",add_bos=True))
print("Canonical tasks/tokens:", len(canonical),len(canonical_ids))
print("Edited canonical token overlap:",sorted(canonical_ids & set(spec)))

unchanged_diffs=[]
for p in pair_roll(17):
    a,b=p["original"],p["edited"]
    if not a["instruction"] or not a["actions_npy"] or not b["actions_npy"]: continue
    tok=set(sp.encode(a["instruction"]+"\n",add_bos=True))
    if tok & set(spec): continue
    x,y=np.load(a["actions_npy"]),np.load(b["actions_npy"])
    unchanged_diffs.append((x.shape==y.shape and np.array_equal(x,y),a["suite"],a["id"]))
print("Untargeted pass17 exact full-action matches:",Counter(r[0] for r in unchanged_diffs))
print("Validation batch sizes:",[len(b) for b in manifest])
print("Validation/dev overlap:",
      len({(r["suite"],r["id"]) for b in manifest for r in b} &
          {(r["suite"],r["id"]) for k in (1,2,4,13,14,15,16,17) for r in read_roll(k)}))
val_requests=[
    {"type":"rollout","model":"both",
     "episodes":[{"suite":r["suite"],"id":r["id"],"seed":23} for r in batch]}
    for batch in manifest
]
(W/"validation_requests.json").write_text(json.dumps(val_requests,indent=1))
print("\nVALIDATION REQUEST 1:")
print(json.dumps(val_requests[0],separators=(",",":")))
```
