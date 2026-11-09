The expanded edit produces 11 recoveries and no regressions on the broader diagnostic sample. I will inspect the remaining failures and the exact changed cases, then save this candidate before testing any more speculative generalization to ambiguous descriptive nouns.
```python
print("Candidate B valid-pair result:",paired_summary(10,True))
p10=paired_records(read_roll(10))
print("\nRemaining valid failures:")
for p in p10.values():
    r=p.get("edited")
    if r and r.get("termination") is not None and not r["success"]:
        print(r["suite"],r["id"],r["instruction"])
print("\nEdited files:",[(str(p.relative_to(W)),round(p.stat().st_size/2**20,1)) for p in (W/"edited").rglob("*") if p.is_file()][:10])
torch.save({"rows":candidate_rows,"strengths":strengths_B,"aliases":candidate_aliases},W/"best_B_recipe.pt")
if not (W/"best_B").exists(): shutil.copytree(W/"edited",W/"best_B")
# Compact coverage check against observed prompts; this is a token lookup, not a forward pass.
known_orig=r1+r2+r5
for group in ["literal","language"]:
    rr=[r for r in known_orig if (r["category"]!="Language Instructions")== (group=="literal")]
    hits=[r for r in rr if set(sp.encode(r["instruction"]+"\n")) & set(candidate_rows)]
    print(group, "known prompts",len(rr),"edited-row hits",len(hits))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate B (pass 10)\n"
            "Diagnostic hard-language paired: 11 recoveries / 0 regressions, listed +8.6 pp SE2.5 (n128); "
            +str(paired_summary(10))+". Runtime nulls verified as symmetric CUDA OOM, excluded from valid stats. "
            "Saved best_B/ and best_B_recipe.pt. Protecting canonical tokens gives exact literal-prompt invariance.\n")
# Plan disjoint confirmation sets now; do not inspect their instruction strings.
diagnostic_keys={(r["suite"],r["id"]) for r in r1+r5}
remaining=[v for v in lang if (v["suite"],v["id"]) not in diagnostic_keys]
rng=np.random.default_rng(80517)
remaining=[remaining[i] for i in rng.permutation(len(remaining))]
confirm_sets=[remaining[i:i+128] for i in range(0,len(remaining),128)]
for j,vs in enumerate(confirm_sets):
    eps=[{"suite":v["suite"],"id":v["id"],"seed":7} for v in vs]
    (W/f"confirm_{j+1}_episodes.json").write_text(json.dumps(eps))
print("Reserved disjoint confirmation sizes:",[len(vs) for vs in confirm_sets])
```
