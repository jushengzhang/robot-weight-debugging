The backend has not accepted a pass yet. I will prepare paired-outcome statistics and then give it a longer startup interval. The acceptance criterion will use paired gains and losses—not two independent success-rate error bars—and repeated seeds will be clustered by variant when results are pooled.

```python
def paired_binary_stats(original, changed):
    a=np.asarray(original,dtype=float)
    b=np.asarray(changed,dtype=float)
    assert a.shape==b.shape and a.ndim==1
    n=len(a)
    d=b-a
    se=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else float("nan")
    return {
        "n":n,"original_pct":100*float(a.mean()) if n else float("nan"),
        "edited_pct":100*float(b.mean()) if n else float("nan"),
        "gain_pp":100*float(d.mean()) if n else float("nan"),
        "paired_se_pp":100*se,
        "wins":int(np.sum(d>0)),"losses":int(np.sum(d<0)),
        "ci95_pp":[100*(float(d.mean())-1.96*se),100*(float(d.mean())+1.96*se)] if n else [],
    }

def clustered_gain_stats(original, changed, cluster_ids):
    a=np.asarray(original,dtype=float)
    b=np.asarray(changed,dtype=float)
    d=b-a
    groups=defaultdict(list)
    for ci,di in zip(cluster_ids,d):
        groups[str(ci)].append(float(di))
    n=len(d);g=len(groups)
    mean=float(d.mean())
    ss=sum((sum(ds)-len(ds)*mean)**2 for ds in groups.values())
    se=math.sqrt(g/(g-1)*ss/(n*n)) if g>1 else float("nan")
    return {"n":n,"clusters":g,"gain_pp":100*mean,"clustered_se_pp":100*se}

print("Paired-statistic sanity check:",
      paired_binary_stats([0,0,1,1],[1,0,1,1]))
print("Waiting for the simulator; no edits are being applied.")
time.sleep(300)
note("- Paired and variant-clustered uncertainty utilities prepared. Waited an additional 300 s for backend startup; still no edits.")
print("Ready to retry; charged passes remain 0.")
```
