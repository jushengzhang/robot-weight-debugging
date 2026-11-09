The first third-seed batch has two regressions and no repairs, a change of -1.6 points. I’ll include that result fully, check whether the affected instructions contain edited tokens, and continue the remaining preselected batches without retuning. This tests how reliably the earlier gain transfers across rollout seeds.
```python
p48 = paired_rows(48)
assert set(p48) == {(e["suite"], e["id"], e["seed"]) for e in requests_final[0]["episodes"]}
disc48 = []
for key, pair in p48.items():
    if bool(pair["original"]["success"]) != bool(pair["edited"]["success"]):
        prompt = pair["original"]["instruction"]
        hits = sorted(set(sp.encode(prompt+"\n", add_bos=True)) & row_set)
        history = {}
        for seed in [23, 37]:
            old = language.get((key[0], key[1], seed))
            if old:
                history[str(seed)] = [old["original"]["success"], old["edited"]["success"]]
        disc48.append({
            "suite": key[0], "id": key[1],
            "original": pair["original"]["success"], "edited": pair["edited"]["success"],
            "instruction": prompt, "edited_tokens": [ascii_piece(i) for i in hits],
            "earlier_seed_outcomes_orig_edit": history
        })
for key, pair in p48.items():
    assert key not in language
    language[key] = pair
    raw_language[key] = pair
print("Pass48 discordances:", json.dumps(disc48, indent=2, ensure_ascii=True))
print("Cumulative reconciled validation:", json.dumps(pair_stats(language), separators=(",", ":")))

# Distinguish actual canonical prompts from task-name metadata tokenization.
actual_canonical_ids = set(
    t for prompt in known_audit_prompts
    for t in sp.encode(prompt.strip().replace("_", " ").replace("\n", " ")+"\n", add_bos=True)
)
assert not actual_canonical_ids & row_set
integrity["observed_canonical_instructions"] = len(known_audit_prompts)
integrity["observed_canonical_token_ids"] = len(actual_canonical_ids)
integrity["protected_metadata_token_ids"] = integrity.pop("protected_token_ids")
(W/"final_integrity.json").write_text(json.dumps(integrity, indent=2))
print("Actual canonical prompts/token IDs:", len(known_audit_prompts), len(actual_canonical_ids),
      "; metadata-derived protected IDs:", len(protected_ids))
report["integrity"] = integrity
report["validation_after48"] = pair_stats(language)
report["validation_raw_after48"] = pair_stats(raw_language)
report["seed73_after48"] = pair_stats(p48)
report["pending_passes"] = [49, 50]
(W/"final_report.json").write_text(json.dumps(report, indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPass48 seed73 reserved batch1:108->106/128,0 repairs/2 regressions, no errors. "
            "No retuning. Discordances="+json.dumps(disc48)+
            "\nCumulative validation after48="+json.dumps(pair_stats(language))+
            f"\nCanonical integrity clarification: actual observed prompts={len(known_audit_prompts)}, "
            f"token IDs={len(actual_canonical_ids)}; metadata task-name strings give89 protected IDs. "
            "Both sets disjoint from edited rows. Final checkpoint unchanged.\n")
print("PASS49 REQUEST:", json.dumps(requests_final[1], separators=(",", ":")))
```
