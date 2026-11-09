The probe failed because the forward service ran out of GPU memory; it did not consume a pass. While it clears, I will audit possible synonym edits against the protected literal-instruction vocabulary and their diagnostic exposure counts. Ambiguous terms will be kept separate from direct synonyms.

```python
direct_alias_candidates = {
    "basin":"bowl", "cupboard":"cabinet", "cooker":"stove", "cooktop":"stove",
    "compartment":"drawer", "central":"middle", "midpoint":"middle",
    "forefront":"front", "framework":"rack", "vessel":"bowl",
    "rounded":"bowl", "flat":"plate", "furniture":"cabinet",
    "fermented":"wine", "grape":"wine", "grapes":"wine", "merlot":"wine",
    "alcoholic":"wine", "dairy":"cheese", "spreadable":"cream",
    "citrus":"orange", "woven":"basket", "ceramic":"mug", "drinking":"mug",
    "heating":"microwave", "appliance":"stove", "container":"bottle"
}
diag_token_sets=[set(sp.encode(r["instruction"],out_type=int)) for r in diag_records]
for s,t in direct_alias_candidates.items():
    si,ti=word_ids(s),word_ids(t)
    hits=[r for r,toks in zip(diag_records,diag_token_sets) if len(si)==1 and si[0] in toks]
    print(f"{s:12s}->{t:10s}",si,ti,"PROTECTED" if any(i in canonical_ids for i in si) else "",
          "exposures",len(hits),"orig failures",sum(not r["success"] for r in hits))
(W/"alias_candidate_audit.json").write_text(json.dumps(direct_alias_candidates,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nAttempted basin-only probe failed with forward-service GPU OOM; did not count. "
            "Null rollout records have no instruction/actions and are missing in both arms. "
            "Valid basin runs: alpha0:0/5,alpha0.5:2/5,alpha1:5/5. "
            "Initial action RMS is not monotone (0.09044,0.03218,0.03768), though both doses improve it.\n")
time.sleep(45)
print("Audit complete; ready to retry smaller probe.")
```
