The seven-row candidate gives two rescues and no regressions, but its gain is too small to establish broad robustness. I will next test descriptive object cues separately, at half strength first. I will exclude ambiguous words such as “container,” “vessel,” “ceramic,” and “heating,” and continue protecting every literal-task token.

```python
def paired_records(k, valid_only=True):
    rr=read_roll(k)
    groups=defaultdict(dict)
    for r in rr:
        groups[(r["suite"],r["id"],r["seed"])][r["model"]]=r
    pairs=[]
    for key,d in groups.items():
        if "original" not in d or "edited" not in d:
            continue
        a,b=d["original"],d["edited"]
        if valid_only and (a.get("termination") is None or b.get("termination") is None):
            continue
        pairs.append((key,a,b))
    return pairs
def paired_summary(k, valid_only=True):
    ps=paired_records(k,valid_only)
    d=np.array([int(b["success"])-int(a["success"]) for _,a,b in ps],dtype=float)
    return {"n":len(ps),"original":sum(a["success"] for _,a,b in ps),
            "edited":sum(b["success"] for _,a,b in ps),
            "gain_pp":round(100*d.mean(),2) if len(d) else None,
            "se_pp":round(100*d.std(ddof=1)/np.sqrt(len(d)),2) if len(d)>1 else None,
            "rescues":int((d==1).sum()),"regressions":int((d==-1).sum())}
print("pass10 valid:",paired_summary(10))
for key,a,b in paired_records(10):
    if a["success"]!=b["success"]:
        print("changed",key,a["success"],"->",b["success"],a["instruction"])
descriptor_aliases={
    "rounded":"bowl","flat":"plate","furniture":"cabinet",
    "fermented":"wine","grape":"wine","grapes":"wine","alcoholic":"wine",
    "dairy":"cheese","citrus":"orange","woven":"basket"
}
def apply_alias_groups(groups,tag):
    E=E0.clone()
    meta=[]
    used=set()
    for mapping,alpha in groups:
        for source,target in mapping.items():
            si,ti=word_ids(source),word_ids(target)
            assert len(si)==len(ti)==1,(source,si,target,ti)
            assert si[0] not in canonical_ids and si[0] not in used,source
            used.add(si[0])
            E[si[0]]=(1-alpha)*E0[si[0]]+alpha*E0[ti[0]]
            meta.append({"source":source,"target":target,"source_id":si[0],"target_id":ti[0],"alpha":alpha})
    save_edit(emb_name,E)
    del E
    (W/f"{tag}.json").write_text(json.dumps(meta,indent=2))
    return meta
candidate_half_meta=apply_alias_groups([(direct_aliases,1.0),(descriptor_aliases,0.5)],"candidate_descriptor_half")
affected_ids={m["source_id"] for m in candidate_half_meta}
pilot_records=[r for r,toks in zip(diag_records,diag_token_sets) if affected_ids&toks]
control_records=[]
for suite in ["libero_goal","libero_spatial","libero_object","libero_10"]:
    control_records.extend([r for r,toks in zip(diag_records,diag_token_sets)
                            if r["suite"]==suite and not (affected_ids&toks)][:2])
pilot_records+=control_records
pilot_eps=[{"suite":r["suite"],"id":r["id"],"seed":7} for r in pilot_records]
(W/"pilot_episodes.json").write_text(json.dumps(pilot_eps))
print("dose panel:",len(pilot_eps),"affected",len(pilot_eps)-len(control_records))
print(json.dumps(pilot_eps,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nPass10 direct7 candidate: requested126 paired episodes,36 missing both arms; "
            f"valid summary {paired_summary(10)}. Two rescues, no regressions. "
            "Next candidate adds10 descriptive-cue rows at alpha0.5; ambiguous terms excluded. "
            "Exposure-selected diagnostic dose panel saved in pilot_episodes.json.\n")
```
