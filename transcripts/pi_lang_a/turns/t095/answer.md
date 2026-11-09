The replication gained 3.28 points (paired SE 0.83). Across both seeds, the gain is 2.95 points with a variant-clustered SE of 0.71. No unused practice language variants remain, so I will keep that as the primary result. I will audit the exact weight changes and reserve the last tests for all 204 difficulty-4/5 validation variants at a third seed, plus fresh safety samples.
```python
import hashlib, gc
emb_name = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
assert list(edited()) == [emb_name]
cur_emb = load(emb_name)
orig_emb = load_original(emb_name)
assert tuple(cur_emb.shape) == tuple(orig_emb.shape) == (257152, 2048)

changed_rows, changed_values = [], 0
for start in range(0, cur_emb.shape[0], 2048):
    stop = min(start + 2048, cur_emb.shape[0])
    diff = cur_emb[start:stop].to(torch.bfloat16) != orig_emb[start:stop].to(torch.bfloat16)
    changed_rows.extend((torch.where(diff.any(dim=1))[0] + start).tolist())
    changed_values += int(diff.sum().item())
assert set(changed_rows) == source_ids
assert all(torch.equal(cur_emb[a["source_id"]].to(torch.bfloat16),
                       orig_emb[a["target_id"]].to(torch.bfloat16)) for a in aliases18)
assert torch.isfinite(cur_emb).all()

canonical_texts = sorted({r["instruction"] for k in range(35, 41) for r in read_roll(k)
                         if r["model"] == "original" and r.get("instruction")})
protected_ids = set(t for s in canonical_texts for t in prompt_ids(s))
assert not (protected_ids & source_ids)
audit_weights = {
    "edited_tensor": emb_name, "shape": list(cur_emb.shape),
    "changed_rows": changed_rows, "changed_scalar_values_at_bf16": changed_values,
    "all_18_rows_equal_original_donor_rows_at_bf16": True,
    "canonical_instruction_count": len(canonical_texts),
    "protected_token_count": len(protected_ids),
    "protected_token_overlap": 0,
    "changed_rows_sha256": hashlib.sha256(
        cur_emb[sorted(source_ids)].to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    ).hexdigest()
}
(W / "final_weight_audit45.json").write_text(json.dumps(audit_weights, indent=2))
print("Weight audit:", json.dumps(audit_weights, separators=(",", ":")))
del cur_emb, orig_emb
gc.collect()

# Outcome-independent secondary check: every D4/D5 variant in the fixed validation set.
validation_ids = sorted({(s, v) for s, v, _ in first_expected})
hard_ids = [key for key in validation_ids if int(variant_meta[key]["difficulty"]) in (4, 5)]
assert len(hard_ids) == 204
rng = np.random.default_rng(7391)
hard_ids = [hard_ids[i] for i in rng.permutation(len(hard_ids))]
hard_third = [{"suite": s, "id": vid, "seed": 7000 + i}
              for i, (s, vid) in enumerate(hard_ids)]
for k, eps in [(47, hard_third[:102]), (48, hard_third[102:])]:
    (W / f"request{k}.json").write_text(json.dumps(
        {"type": "rollout", "model": "both", "episodes": eps}, separators=(",", ":")))

# Fresh safety variants, selected from metadata only, before viewing outcomes.
rng_safety = np.random.default_rng(8947)
safety_new = {}
for category, n in [("Light Conditions", 128), ("Background Textures", 64), ("Sensor Noise", 64)]:
    pool = sorted(key for key, r in variant_meta.items()
                  if r.get("category") == category and key not in all_used)
    assert len(pool) >= n, (category, len(pool), n)
    safety_new[category] = [pool[i] for i in rng_safety.permutation(len(pool))[:n]]
eps49 = [{"suite": s, "id": vid, "seed": 8000 + i}
         for i, (s, vid) in enumerate(safety_new["Light Conditions"])]
eps50 = [{"suite": s, "id": vid, "seed": 8400 + i}
         for i, (s, vid) in enumerate(safety_new["Background Textures"] + safety_new["Sensor Noise"])]
for k, eps in [(49, eps49), (50, eps50)]:
    assert all((e["suite"], e["id"]) in allowed for e in eps)
    (W / f"request{k}.json").write_text(json.dumps(
        {"type": "rollout", "model": "both", "episodes": eps}, separators=(",", ":")))

print("Final tests precommitted:",
      {"46": "40 canonical fixed-input probes", "47+48": "204 D4/D5 third-seed pairs",
       "49": "128 fresh lighting pairs", "50": "64 fresh background + 64 fresh sensor pairs"})
print("Prior lighting input audit:")
print((W / "light37_input_audit.json").read_text()[:1600])

notes = (W / "notes.md").read_text()
notes += ("\nFinal tests precommitted before outcomes:46 canonical40 fixed-input replay; "
          "47/48 all204 non-diagnostic D4/D5 variants at third seeds7000+ (secondary, not pooled "
          "with primary broad result);49 fresh128Light;50 fresh64Background+64Sensor. "
          "No unused practice language variants remain. Full weight audit final_weight_audit45.json "
          "confirms exactly18 donor-row substitutions and zero canonical-token overlap.\n")
(W / "notes.md").write_text(notes)
print("Candidate unchanged.")
```
