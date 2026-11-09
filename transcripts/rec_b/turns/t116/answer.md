H15 gained 3 episodes on validation B, but that exactly offsets its 3-episode loss on validation A: there is no pooled success-rate gain. I will not retain it on the strength of the favorable L3 subgroup. I will now restore the frozen routing-softening candidate and test it on the same validation-B panel.
```python
def valid_pairs(k):
    ps=paired_table(k)
    return [(key,o,e) for key,o,e in ps
            if o.get("termination") in ["success","max_steps"]
            and e.get("termination") in ["success","max_steps"]]

def pooled_stats(passes):
    pairs=[p for k in passes for p in valid_pairs(k)]
    d=np.array([int(e["success"])-int(o["success"]) for _,o,e in pairs],dtype=float)
    return {"n":len(d),"original":sum(o["success"] for _,o,e in pairs),
            "edited":sum(e["success"] for _,o,e in pairs),
            "diff_pp":round(float(d.mean()*100),3),
            "se_pp":round(float(d.std(ddof=1)/math.sqrt(len(d))*100),3),
            "rescues":int((d>0).sum()),"regressions":int((d<0).sum())}
print("H15 validation A+B:",pooled_stats([26,45]))
print("Scene errors:",[(r["id"],r["grip"],r.get("termination")) for r in read_jsonl(W/"roll_45.jsonl")
                       if r.get("termination") not in ["success","max_steps"]][:8])

restore_snapshot("candidate_I065")
print("Frozen I065 restored:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## H15 independent replication, pass45\n"
            "ValidationB60/126->63/126,+2.4+/-2.9pp,8 rescues/5 regressions; "
            "two paired episodes excluded for scene errors. "
            "Pooled A+B133/254->133/254:zero net gain. H15 does not qualify. "
            "Frozen I065 restored for second predeclared replication.\n")
```
