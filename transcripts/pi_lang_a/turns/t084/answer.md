Across the six safety categories, the observed changes range from 0 to −1.56 percentage points. I will check whether any of those prompts uses an edited token, then print the first precommitted replication request exactly as saved to avoid transcription errors.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH) / "paligemma_tokenizer.model"))
aliases18 = json.loads((W / "candidate18_frozen.json").read_text())
source_ids = {a["source_id"] for a in aliases18}
def prompt_ids(s):
    return sp.encode(s.strip().lower().replace("_", " ") + "\n", add_bos=True)

exposure40 = {}
for k in range(35, 41):
    rr = [r for r in read_roll(k) if r["model"] == "original"]
    hits = [(r["suite"], r["id"], r["seed"]) for r in rr
            if r.get("instruction") and source_ids.intersection(prompt_ids(r["instruction"]))]
    exposure40[rr[0]["category"]] = {"n": len(rr), "exposed": len(hits), "episodes": hits}
print("Safety edited-token exposure:", json.dumps(exposure40, separators=(",", ":")))
(W / "final18_safety40_exposure.json").write_text(json.dumps(exposure40, indent=2))

practice = json.loads((Path(BENCH) / "practice_variants.json").read_text())
def find_variant_ids(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from find_variant_ids(x)
    elif isinstance(obj, dict):
        if "suite" in obj and "id" in obj:
            yield (obj["suite"], int(obj["id"]))
        else:
            for v in obj.values():
                if isinstance(v, (list, dict)):
                    yield from find_variant_ids(v)
allowed = set(find_variant_ids(practice))
assert allowed, "Inspect manifest schema before proceeding"
assert all((e["suite"], e["id"]) in allowed for b in rep_batches for e in b)
assert all((e["suite"], e["id"]) in allowed for e in repair)
print("Replication: all 610 episodes and repair are practice variants.")
print("REQUEST41")
print((W / "request41.json").read_text())
```
