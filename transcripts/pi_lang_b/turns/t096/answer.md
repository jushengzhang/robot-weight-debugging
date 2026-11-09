All six new difficulty-4/5 audits have identical success outcomes: 768 paired episodes with no repairs or regressions. I’ll now combine them with the earlier safety tests without double-counting repeats, verify that only the intended 20 rows differ from the original, and prepare the final language-validation batches.
```python
import hashlib, math
from collections import Counter

def pair_stats(ps):
    items = list(ps.items()) if isinstance(ps, dict) else list(enumerate(ps))
    n = len(items)
    orig = np.array([bool(p["original"].get("success")) for _, p in items], dtype=float)
    edit = np.array([bool(p["edited"].get("success")) for _, p in items], dtype=float)
    d = edit-orig
    clusters = defaultdict(float)
    for (key, p), di in zip(items, d):
        rr = p["original"]
        clusters[(rr["suite"], rr["id"])] += float(di-d.mean())
    g = len(clusters)
    cse = math.sqrt(g/(g-1)*sum(v*v for v in clusters.values()))/n if g > 1 else 0.0
    errors = {arm: sum(p[arm].get("success") is None or
                       str(p[arm].get("termination", "")).startswith("error")
                       for _, p in items) for arm in ["original", "edited"]}
    return {
        "n": n, "variants": g, "original": int(orig.sum()), "edited": int(edit.sum()),
        "fixes": int((d > 0).sum()), "regressions": int((d < 0).sum()),
        "diff_pp": round(float(100*d.mean()), 4),
        "paired_se_pp": round(float(100*d.std(ddof=1)/np.sqrt(n)), 4) if n > 1 else 0.0,
        "cluster_se_pp": round(100*cse, 4),
        "cluster_95ci_pp": [round(100*(float(d.mean())-1.96*cse), 4),
                            round(100*(float(d.mean())+1.96*cse), 4)],
        "errors": errors
    }

audit_passes = list(range(25, 32)) + list(range(42, 48))
audit_unique = defaultdict(dict)
audit_repeats = Counter()
all_audit_prompts = set()
for k in audit_passes:
    for key, pair in paired_rows(k).items():
        cat = pair["original"]["category"]
        all_audit_prompts.add(pair["original"]["instruction"])
        if key not in audit_unique[cat]:
            audit_unique[cat][key] = pair  # earliest trial retained, not best outcome
        else:
            audit_repeats[cat] += 1
safety_summary = {cat: pair_stats(ps) for cat, ps in audit_unique.items()}
print("SAFETY (unique variant/seed pairs; earliest trial retained):")
for cat, st in safety_summary.items():
    print(cat, json.dumps(st, separators=(",", ":")))
print("Removed repeated trials:", dict(audit_repeats))

raw_language = {}
for k in [18, 19, 20, 21, 33, 34, 35, 36, 37]:
    for key, pair in paired_rows(k).items():
        assert key not in raw_language
        raw_language[key] = pair
language = dict(raw_language)
retry_pairs = paired_rows(24)
assert len(retry_pairs) == 25
for key, pair in retry_pairs.items():
    assert key in language
    assert any(r.get("success") is None or str(r.get("termination", "")).startswith("error")
               for r in language[key].values())
    language[key] = pair
assert not any((k[0], k[1]) in dev_set for k in language)
print("LANGUAGE before final3 batches, raw:", json.dumps(pair_stats(raw_language), separators=(",", ":")))
print("LANGUAGE with prespecified same-seed error retries:",
      json.dumps(pair_stats(language), separators=(",", ":")))

canonical_texts = {r["name"].split("_language_")[0].replace("_", " ")
                   for batch in manifest for r in batch}
protected_ids = set(t for text in canonical_texts for t in sp.encode(text+"\n", add_bos=True))
assert not protected_ids.intersection(row_set)
audit_triggers = [(p, sorted(set(sp.encode(p.strip().replace("_", " ").replace("\n", " ")+"\n",
                                           add_bos=True)) & row_set))
                  for p in all_audit_prompts]
assert not any(hit for _, hit in audit_triggers)
assert edited() == [ENAME]

cur, orig = load(ENAME), load_original(ENAME)
changed_rows, changed_coefficients = [], 0
for start in range(0, cur.shape[0], 4096):
    diff = cur[start:start+4096] != orig[start:start+4096]
    changed_rows.extend((torch.nonzero(diff.any(dim=1)).flatten()+start).tolist())
    changed_coefficients += int(torch.count_nonzero(diff))
assert changed_rows == row_ids
assert changed_coefficients == 40822
assert torch.equal(cur[row_idx], torch.from_numpy(final_rows))
assert torch.isfinite(cur[row_idx]).all()
assert torch.equal(cur[row_idx].to(torch.bfloat16).float(), cur[row_idx])
del cur, orig, diff
gc.collect()

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(16*1024*1024)
            if not block:
                return h.hexdigest()
            h.update(block)
artifact_path = W/"edited"/(ENAME+".npy")
checkpoint_sha = sha256_file(final_path)
artifact_sha = sha256_file(artifact_path)
assert checkpoint_sha == artifact_sha
integrity = {
    "tensor": ENAME, "rows_changed": changed_rows,
    "coefficients_changed": changed_coefficients,
    "canonical_tasks": len(canonical_texts), "protected_token_ids": len(protected_ids),
    "other_tensors_edited": 0, "nonlanguage_prompts_using_edited_rows": 0,
    "exact_checkpoint_sha256": artifact_sha, "bf16_target_rows_exact": True
}
(W/"final_integrity.json").write_text(json.dumps(integrity, indent=2))
report = {"integrity": integrity, "safety": safety_summary,
          "validation_raw_before_last3": pair_stats(raw_language),
          "validation_reconciled_before_last3": pair_stats(language),
          "error_retry_pass": 24, "pending_passes": [48,49,50]}
(W/"final_report.json").write_text(json.dumps(report, indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPasses42-47 difficult nonlanguage audits: 768 paired trials, zero outcome discordances; "
            "Camera73/128, Robot78, Light113, Background106, Sensor101, Layout88 each arm. "
            "Combined deduplicated safety: "+json.dumps(safety_summary)+
            "\nFinal artifact integrity: exactly20 rows/40822 coefficients in embedding table differ; "
            f"{len(protected_ids)} canonical token IDs untouched, all other tensors unchanged. "
            "All audited nonlanguage instructions exclude edited rows. Artifact SHA256="+artifact_sha+
            ". Matches lex_dest_exact checkpoint byte-for-byte.\n")
print("INTEGRITY:", len(changed_rows), "rows;", changed_coefficients, "coefficients;",
      len(protected_ids), "protected canonical token IDs; checkpoint SHA256", artifact_sha[:16])
print("PASS48 REQUEST:", json.dumps(requests73[0], separators=(",", ":")))
```
