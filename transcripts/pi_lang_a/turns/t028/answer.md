The single-row tests support a joint task-code problem: basin alone gives no completed rescues, cupboard alone gives one, and both rows together give five in the earlier valid runs. I will extend the same intervention to a small set of direct synonyms, while keeping descriptive and ambiguous words out of this first broader candidate.

```python
direct_aliases={
    "basin":"bowl","cupboard":"cabinet","cooker":"stove","forefront":"front",
    "central":"middle","midpoint":"middle","framework":"rack"
}
apply_aliases(direct_aliases,1.0,"direct_aliases_full")
# Fix a diagnostic evaluation panel before evaluating the broader candidates.
goal_diag=[r for r in diag_records if r["suite"]=="libero_goal"]
other_diag=[]
for suite in ["libero_spatial","libero_object","libero_10"]:
    rs=[r for r in diag_records if r["suite"]==suite]
    rs.sort(key=lambda r:(r["success"],r["id"]))  # retain every original failure
    other_diag.extend(rs[:12])
tune_records=goal_diag+other_diag
tune_eps=[{"suite":r["suite"],"id":r["id"],"seed":7} for r in tune_records]
(W/"tune_episodes.json").write_text(json.dumps(tune_eps))
print("fixed diagnostic panel:",len(tune_eps),"original successes:",sum(r["success"] for r in tune_records))
print(json.dumps(tune_eps,separators=(",",":")))
print("Descriptor contexts to audit:")
for word in ["rounded","furniture","heating","ceramic"]:
    hits=[r for r,toks in zip(diag_records,diag_token_sets) if word_ids(word)[0] in toks]
    print(word,[(r["suite"],r["id"],r["success"],r["instruction"]) for r in hits])
with (W/"notes.md").open("a") as f:
    f.write("\nPass9 cupboard-only:1 basin rescue out of4 valid basin cases (3/8 requests missing both arms). "
            "Joint row intervention substantially stronger than either alone. "
            "Candidate D:7 direct synonym rows at full dose; canonical 74 token IDs protected. "
            "Fixed diagnostic panel saved in tune_episodes.json.\n")
```
