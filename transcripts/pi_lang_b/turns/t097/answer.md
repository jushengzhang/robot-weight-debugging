The combined safety audits show no observed decline larger than 0.85 percentage points, but three earlier camera pairs contain infrastructure errors. The bookkeeping code also encountered a missing instruction in an error record. I’ll keep those outcomes in the raw statistics, finish the weight-integrity check, and reserve three slots in the last pass to retry every camera error pair.
```python
# Missing prompts belong to infrastructure-error records, not to valid instructions.
known_audit_prompts = {p for p in all_audit_prompts if isinstance(p, str)}
audit_triggers = {
    p: sorted(set(sp.encode(p.strip().replace("_", " ").replace("\n", " ")+"\n",
                           add_bos=True)) & row_set)
    for p in known_audit_prompts
}
assert not any(audit_triggers.values())
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
    "other_tensors_edited": 0, "known_nonlanguage_prompts_using_edited_rows": 0,
    "exact_checkpoint_sha256": artifact_sha, "bf16_target_rows_exact": True
}
(W/"final_integrity.json").write_text(json.dumps(integrity, indent=2))

def has_error(pair):
    return any(r.get("success") is None or str(r.get("termination", "")).startswith("error")
               for r in pair.values())

camera_error_keys = [key for key, pair in audit_unique["Camera Viewpoints"].items() if has_error(pair)]
assert len(camera_error_keys) == 3
camera_retry_episodes = [{"suite": k[0], "id": k[1], "seed": k[2]} for k in camera_error_keys]
requests_final = json.loads(json.dumps(requests73))
omitted_final_language = requests_final[2]["episodes"][-len(camera_retry_episodes):]
requests_final[2]["episodes"] = requests_final[2]["episodes"][:-len(camera_retry_episodes)] + camera_retry_episodes
(W/"final_three_requests.json").write_text(json.dumps(requests_final, indent=2))
(W/"camera_error_retry_manifest.json").write_text(json.dumps({
    "episodes": camera_retry_episodes,
    "language_slots_replaced_by_position_not_outcome": omitted_final_language,
    "reason": "Retry all infrastructure-error camera pairs; preserve raw statistics separately"
}, indent=2))

# Zero observed discordances do not imply a zero-width population confidence interval.
for st in safety_summary.values():
    if st["fixes"]+st["regressions"] == 0:
        bound = 100*(1-0.05**(1/st["n"]))
        st["zero_discordance_conservative_95_abs_difference_bound_pp"] = round(bound, 4)

report = {
    "integrity": integrity, "safety_raw_deduplicated": safety_summary,
    "validation_raw_before_last3": pair_stats(raw_language),
    "validation_reconciled_before_last3": pair_stats(language),
    "language_error_retry_pass": 24, "pending_passes": [48,49,50],
    "pending_language_trials": 381, "pending_camera_error_retries": 3,
    "selection_frozen_before_pass18": True
}
(W/"final_report.json").write_text(json.dumps(report, indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPasses42-47 difficult nonlanguage audits: 768 paired trials, zero outcome discordances; "
            "Camera73/128, Robot78, Light113, Background106, Sensor101, Layout88 each arm. "
            "Combined deduplicated safety: "+json.dumps(safety_summary)+
            "\nFinal artifact integrity: exactly20 rows/40822 coefficients in embedding table differ; "
            f"{len(protected_ids)} canonical token IDs untouched, all other tensors unchanged. "
            "All observed nonlanguage instructions exclude edited rows. Artifact SHA256="+artifact_sha+
            ". Matches lex_dest_exact checkpoint byte-for-byte.\n"
            "Final3 requests now in final_three_requests.json: first2 remain128 language each; "
            "third uses first125 language variants +all3 old camera infrastructure-error pairs. "
            "Last3 language slots dropped strictly by manifest position before seed73 outcomes seen, "
            "not by task/outcome. Total381 new language trials, explicit seed73. Raw and retried camera "
            "results will be reported separately; no errors silently excluded.\n")
print("INTEGRITY:", len(changed_rows), "rows;", changed_coefficients, "coefficients;",
      len(protected_ids), "protected canonical token IDs; checkpoint SHA256", artifact_sha[:16])
print("Camera retries:", camera_retry_episodes)
print("PASS48 REQUEST:", json.dumps(requests_final[0], separators=(",", ":")))
```
