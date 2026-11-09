The fresh lighting sample gained 2 successes, bringing the combined lighting difference to zero over 256 pairs. I will verify the final weight checksum, save the evidence and submission choice, and load the last request: fresh Background Textures and Sensor Noise episodes.
```python
print("Pass49 audit:", audit_request(49))
safety_rows49 = [r for k in list(range(35, 41)) + [49] for r in read_roll(k)]
safety_summary49 = grouped_stats(safety_rows49, "category")
exposed49 = [ekey(r) for r in read_roll(49) if r["model"] == "original"
             and r.get("instruction") and source_ids.intersection(prompt_ids(r["instruction"]))]
print("Pass49 edited-token exposures:", len(exposed49))
print("Safety through49:")
for category, s in safety_summary49.items():
    print(f"  {category}: {s['original']}->{s['edited']}/{s['n']}, "
          f"{s['gain_pp']:+.3f} pp, SE {s['se_pp']:.3f}")
(W / "final18_safety49_summary.json").write_text(json.dumps(safety_summary49, indent=2))

assert list(edited()) == [emb_name]
active = load(emb_name)
active_hash = hashlib.sha256(
    active[sorted(source_ids)].to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
).hexdigest()
assert active_hash == audit_weights["changed_rows_sha256"]
del active
gc.collect()
print("Submission unchanged: 18-row embedding edit; checksum verified.")

final_choice = {
    "candidate": "frozen18",
    "active_edited_tensors": list(edited()),
    "recipe": "candidate18_frozen.json",
    "backup": "candidate18_weights",
    "weight_audit": "final_weight_audit45.json",
    "changed_rows_sha256": active_hash,
    "primary_language_evidence": "final18_evidence45.json",
    "secondary_hard_language_evidence": "final18_hard_third_seed_summary.json",
    "canonical_invariance": "canonical_probe46_audit.json",
    "safety_through_pass49": "final18_safety49_summary.json",
    "last_safety_request": "request50.json",
    "last_safety_records_when_complete": "roll_50.jsonl",
    "training_used": False,
    "all_non_embedding_tensors_unchanged": True
}
(W / "final_choice.json").write_text(json.dumps(final_choice, indent=2))

notes = (W / "notes.md").read_text()
notes = re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 49/50.", notes)
notes = re.sub(r"Current best validated:[^\n]*",
              "Final chosen submission: frozen candidate18, unchanged since pass28. "
              "Only 18 instruction-embedding rows replaced with original canonical donor rows. "
              "Backup candidate18_weights; recipe candidate18_frozen.json; final_choice.json records evidence.",
              notes, count=1)
notes += (
    "\nPass49 fresh Light120->122/128 (+1.563pp;3rescues1regression), zero source-token exposure. "
    "Combined final18 Light242->242/256. Manifest exact. "
    "Submission checksum reverified before last pass. "
    "Primary language:1068->1104/1220 (+2.951pp, variant-cluster SE0.7146), "
    "42rescues6regressions,610 non-diagnostic variants x2seeds. "
    "Primary D5:127->159/218 (+14.679pp). Independent fresh354 initial confirmation:+2.54pp(SE1.01). "
    "Secondary all204 D4/D5 at third seeds:144->158 (+6.863pp,SE2.028); D5:57->72/109. "
    "Canonical replay40/40 action chunks and1920/1920 activation arrays exact. "
    "Final pass50 precommitted64freshBackground+64freshSensor; retain all outcomes. "
    "Active weights are the chosen final18; no subsequent edits planned. "
    "See roll_50.jsonl for the final safety result when available.\n"
)
(W / "notes.md").write_text(notes)
print("REQUEST50")
print((W / "request50.json").read_text())
```
