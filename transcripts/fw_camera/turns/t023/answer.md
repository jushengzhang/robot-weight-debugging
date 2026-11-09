The fixed panels cover all four camera-parameter families present in the practice list. I will now prepare the paired-effect calculations and uncertainty checks, including clustered estimates for repeated variants. These will be used to decide whether a gain is larger than its sampling error and whether control-category changes stay within the allowed margin.```python
def paired_stats(original, modified):
    a=np.asarray(original,dtype=float)
    b=np.asarray(modified,dtype=float)
    assert a.shape==b.shape and a.ndim==1 and len(a)>0
    d=b-a
    n=len(d)
    se=float(d.std(ddof=1)/np.sqrt(n)) if n>1 else float("nan")
    wins=int(((a==0)&(b==1)).sum())
    losses=int(((a==1)&(b==0)).sum())
    return {"n":n,"original_pct":100*float(a.mean()),"edited_pct":100*float(b.mean()),
            "gain_pp":100*float(d.mean()),"se_pp":100*se,
            "ci95_pp":[100*(float(d.mean())-1.96*se),100*(float(d.mean())+1.96*se)],
            "wins":wins,"losses":losses,"discordant":wins+losses}

def clustered_paired_stats(original,modified,variant_keys,reps=10000,seed=7181):
    base=paired_stats(original,modified)
    dif=np.asarray(modified,dtype=float)-np.asarray(original,dtype=float)
    groups=collections.defaultdict(list)
    for key,d in zip(variant_keys,dif):
        groups[str(key)].append(float(d))
    counts=np.array([len(v) for v in groups.values()])
    totals=np.array([sum(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    ids=rng.integers(len(groups),size=(reps,len(groups)))
    boot=totals[ids].sum(1)/counts[ids].sum(1)
    base["variant_clusters"]=len(groups)
    base["cluster_bootstrap_se_pp"]=100*float(boot.std(ddof=1))
    base["cluster_bootstrap_ci95_pp"]=(100*np.quantile(boot,[0.025,0.975])).tolist()
    return base

def log_pass(k,kind,purpose,edit_label="original",extra=None):
    global ledger
    entry={"pass":k,"type":kind,"purpose":purpose,"edit":edit_label}
    if extra:
        entry.update(extra)
    ledger=[e for e in ledger if e["pass"]!=k]+[entry]
    ledger.sort(key=lambda e:e["pass"])
    (W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))

print("Paired estimator sanity check on synthetic bookkeeping, not model data:")
print(paired_stats([0,0,1,1],[1,0,1,1]))
print("Evaluation rules:")
print("- Discovery outcomes can select a dose; reserved validation cannot.")
print("- Report wins/losses and paired SE, not independent-binomial SE.")
print("- Repeated-seed aggregates will also report variant-clustered uncertainty.")
print("- Test every noncamera category; add power where uncertainty is largest.")
print("Forward passes used:",len(ledger),"; current edits:",list(edited()))
print("Waiting 850 seconds for backend startup.")
time.sleep(850)
print("Wait complete.")
```
