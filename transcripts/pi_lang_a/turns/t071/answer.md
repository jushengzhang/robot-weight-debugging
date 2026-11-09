Across all 354 untouched variants, the final edit gained nine net successes. I will compute the paired uncertainty, save a fixed second-seed replication plan, and then retest the earlier validation set with the final weights to broaden coverage.
```python
fresh_rows=[r for k in (30,31,32) for r in records(k)]
fresh_s,fresh_pairs=paired_summary(fresh_rows)
b=fresh_s["rescues"]; c=fresh_s["regressions"]; ndis=b+c
p_exact=min(1.0,2*sum(math.comb(ndis,i) for i in range(min(b,c)+1))/2**ndis) if ndis else 1.0
print("Fresh354 summary:",fresh_s,"McNemar exact two-sided p:",round(p_exact,6))
for suite in suites:
    print(suite,paired_summary([r for r in fresh_rows if r["suite"]==suite])[0])
(W/"final18_fresh354_summary.json").write_text(json.dumps({"summary":fresh_s,"p_exact":p_exact},indent=2))

# Precommit later replication seeds now, without selecting favorable variants.
firstpass610=confirm_first+confirm_second+confirm_third+batch17+batch18
assert len(firstpass610)==610 and len({(e["suite"],e["id"]) for e in firstpass610})==610
rng_rep=np.random.default_rng(180041)
order=rng_rep.permutation(len(firstpass610))
replicate610=[dict(firstpass610[int(j)],seed=4000+i) for i,j in enumerate(order)]
rep_batches=[replicate610[i:i+128] for i in range(0,len(replicate610),128)]
(W/"final18_replication_batches.json").write_text(json.dumps(rep_batches,indent=2))
print("Precommitted replication sizes:",[len(x) for x in rep_batches])
with (W/"notes.md").open("a") as f:
    f.write("\nPass32 remaining fresh113: original106 edited106,1rescue1regression. FRESH354 final18 aggregate310->319 (+2.54pp,SE1.01),11rescues2regressions,exact McNemar p0.02246. Independent confirmation criterion met; goal29->37/49. No alias changes. Planned next:33/34 rerun request17/request18 panels with frozen18 (total610 unique non-diagnostic language variants);35-40 repeat six-category safety;41-45 precommitted second-seed replication610 in final18_replication_batches.json; remaining passes for mechanistic controls/final invariance. Candidate18_weights is final intended backup.\n")
print("Request33:",json.dumps({"type":"rollout","model":"both","episodes":batch17},separators=(",",":")))
```