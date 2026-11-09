The full-dose result supports the earlier dose-response finding: success rose from 49/80 unedited to 59/80 at half strength and 68/80 at full strength. I will now verify the saved weights and token coverage, then prepare fresh-seed replications on the 460 variants reserved before F was evaluated.
```python
import sentencepiece as spm
torch.set_num_threads(24)
tok = spm.SentencePieceProcessor(model_file=str(Path(BENCH) / "paligemma_tokenizer.model"))
ename = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
recipe = torch.load(W / "best_F_recipe.pt", map_location="cpu", weights_only=False)
rows, strengths = recipe["rows"], recipe["strengths"]
E = load(ename)
E0 = load_original(ename)
B = np.load(W / "best_F_dev" / (ename + ".npy"), mmap_mode="r")
changed = []
same_best = True
for start in range(0, E.shape[0], 4096):
    end = min(start + 4096, E.shape[0])
    changed.extend((torch.where((E[start:end] != E0[start:end]).any(dim=1))[0] + start).tolist())
    same_best &= np.array_equal(E[start:end].numpy(), B[start:end])
assert set(changed) == set(rows), (len(changed), len(rows))
assert same_best
print("WEIGHT AUDIT:", {"shape": list(E.shape), "changed_rows": len(changed),
                        "equal_to_frozen_F": same_best,
                        "fraction_of_embedding_rows": len(changed) / E.shape[0]})
print("ROW RECIPE (source, target(s), strength):")
for i, target in rows.items():
    targets = [target] if isinstance(target, (int, np.integer)) else list(target)
    print(tok.id_to_piece(int(i)).encode("ascii", "backslashreplace").decode(),
          "->", [tok.id_to_piece(int(j)).encode("ascii", "backslashreplace").decode() for j in targets],
          strengths[i])

safety_records = []
for k in range(24, 30):
    safety_records += [json.loads(x) for x in (W / f"roll_{k}.jsonl").read_text().splitlines() if x.strip()]
safety_prompts = sorted(set(r["instruction"] for r in safety_records))
protected = set()
for text in safety_prompts:
    protected.update(tok.encode(text.strip().lower().replace("_", " ") + "\n", add_bos=True))
print("SAFETY PROMPT AUDIT:", {"unique_prompts": len(safety_prompts),
                               "unique_token_ids": len(protected),
                               "edited_token_intersection": sorted(set(changed) & protected)})
assert not (set(changed) & protected)

variants = json.loads((Path(BENCH) / "practice_variants.json").read_text())
print("VARIANT FILE TYPE:", type(variants).__name__)
print("VARIANT SCHEMA:", str(variants[:2] if isinstance(variants, list) else list(variants.items())[:1])[:1200])
for k in range(2, 6):
    eps = json.loads((W / f"confirm_{k}_episodes.json").read_text())
    print("CONFIRM", k, "n=", len(eps), "first=", eps[:2])
print("CONFIRMATION RESULT SCHEMA:", str(json.loads((W / "F_confirmation_results.json").read_text()))[:1000])
print("PROBE INPUT MANIFEST:", (W / "F_controlled_probe_cases.json").read_text()[:1700])
print("FRAME DIRECTORY EXAMPLE:")
for p in sorted((W / "roll_29_frames").iterdir())[:2]:
    print(str(p), "dir=", p.is_dir())
    if p.is_dir():
        print([q.name for q in p.iterdir()][:12])

# Prespecified fresh-seed replication; no changes to F.
replication_plan = {
    "candidate": "frozen_F",
    "variant_lists": [f"confirm_{k}_episodes.json" for k in range(2, 6)],
    "new_seeds": [3101, 6203],
    "analysis": "paired differences; cluster standard errors by (suite,id) across seeds",
    "planned_passes": {
        "35-38": "460 reserved variants, seed3101",
        "39-42": "same460 reserved variants, seed6203",
        "43-48": "128 additional paired episodes per non-language category",
        "49-50": "fixed-input safety and mechanism probes"
    }
}
(W / "F_replication_plan.json").write_text(json.dumps(replication_plan, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Prespecified remaining validation\n"
            "Full F equals best_F_dev elementwise; exactly65 embedding rows changed. "
            "Fresh-seed replication on the460 reserved variants: seeds3101 and6203, "
            "four batches per seed; cluster uncertainty by variant. Then128 additional "
            "paired episodes in each other category, and two fixed-input probes. "
            "No tuning planned from these results.\n")
del B
```
