The value-only edit did not improve task compliance: old-goal failures stayed at 10, and its overall gain was smaller than its standard error. I’m rejecting it. Next I’ll test the same isolated instruction subspace in the key projections, where it can change routing rather than merely amplify values that may not be read.

```python
r4=read_roll(4)
def paired_summary(rows):
    by=collections.defaultdict(dict)
    for r in rows: by[(r["id"],r["seed"])][r["model"]]=r
    out={}
    for p in ("all","object","semantic","task","position","environment"):
        pairs=[v for v in by.values() if "original" in v and "edited" in v
               and (p=="all" or v["original"]["ptype"]==p)]
        if not pairs: continue
        d=np.array([int(v["edited"]["success"])-int(v["original"]["success"]) for v in pairs])
        out[p]={"n":len(d),"delta_pp":round(100*d.mean(),2),
                "se_pp":round(100*d.std(ddof=1)/math.sqrt(len(d)),2) if len(d)>1 else None,
                "wins":int((d>0).sum()),"losses":int((d<0).sum())}
    return out
print("value-only paired:",paired_summary(r4))
print("changed outcomes:",[(eid,v["original"]["ptype"],v["original"]["success"],v["edited"]["success"])
      for eid,v in {
          e["id"]:{r["model"]:r for r in r4 if r["id"]==e["id"]} for e in diag_eps
      }.items() if v["original"]["success"] != v["edited"]["success"]])
clear_edits()
def text_key_edit(alpha=2.0,layers=tuple(range(6,12)),rank=8,reset=True):
    if reset: clear_edits()
    stats=[]
    for l in layers:
        n=f"{VLM}.layers.{l}.self_attn.k_proj.weight"
        w=load_original(n); u=text_bases[l][:,:rank]
        save_edit(n,w+alpha*(w@u)@u.T)
        stats.append((l,round(float((load(n)-w).norm()/w.norm()),4)))
    return stats
print("key edits:",text_key_edit())
snapshot("candidate_textK_mid_a2")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 4: rejected value-only edit\n"
            "44/75 ->45/75 (+1.3 +/-3.0 pp paired SE), task 4/15 unchanged, "
            "old-goal failures 10 unchanged; object 14->12, env7->9, position4->5. "
            "Not evidence for the proposed task repair. Reverted.\n"
            "Next: key-only rank8 instruction-subspace gain alpha=2 in layers6-11, "
            "to test routing rather than value strength.\n")
```
