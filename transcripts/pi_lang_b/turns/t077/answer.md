The second seed yielded seven repairs and one regression. I will combine both seeds with variant-clustered uncertainty and document the regression. Then I will prepare a final untouched-variant check using all 65 remaining practice-language variants, with a second seed for a randomly selected 63 of them.
```python
ps37=[p for k in range(33,37) for p in pair_roll(k)]
print("Seed37 full reserved validation:",paired_stats(ps37))
both_seed_validation=vfinal+ps37
def clustered_stats(ps):
    g=defaultdict(list)
    for p in ps:
        a,b=p["original"],p["edited"]
        g[(a["suite"],a["id"])].append(int(bool(b["success"]))-int(bool(a["success"])))
    x=np.array([np.mean(v) for v in g.values()])
    se=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0
    return {"variants":len(x),"paired_trials":len(ps),
            "diff_pp":round(float(x.mean()*100),3),"cluster_se_pp":round(float(se*100),3),
            "normal_95ci_pp":[round(float((x.mean()-1.96*se)*100),3),
                              round(float((x.mean()+1.96*se)*100),3)]}
print("Two seeds pooled:",paired_stats(both_seed_validation))
print("Variant-clustered:",clustered_stats(both_seed_validation))
for p in ps37:
    if p["original"]["success"] and not p["edited"]["success"]:
        a=p["original"]
        print("Replication regression:",a["suite"],a["id"],a["seed"],repr(a["instruction"]),
              "edited tokens",sorted(set(sp.encode(a["instruction"]+"\n",add_bos=True)) & set(spec)))

fresh_language=json.loads((W/"final_fresh_language_manifest.json").read_text())
rng=np.random.default_rng(305)
second_fresh=[fresh_language[i] for i in rng.choice(len(fresh_language),63,replace=False)]
fresh_req={"type":"rollout","model":"both",
           "episodes":[{"suite":r["suite"],"id":r["id"],"seed":61} for r in fresh_language]+
                      [{"suite":r["suite"],"id":r["id"],"seed":67} for r in second_fresh]}
(W/"final_fresh_language_request.json").write_text(json.dumps(fresh_req,indent=1))
print("Fresh variant mix:",dict(Counter((r["suite"],r["difficulty"]) for r in fresh_language)))
with (W/"notes.md").open("a") as f:
    f.write("\nPasses33-36 exact-source artifact, reserved512 variants at second seed37: "
            +str({k:v for k,v in paired_stats(ps37).items() if k!="errors"})
            +". No infrastructure errors. Two-seed validation "
            +str(clustered_stats(both_seed_validation))+".\n"
            "Next37: all65 remaining untouched language variants seed61 + random63 of these seed67 (128 pairs). "
            "Then temporary norm/direction and destination-dose controls (38-41), RESTORE checkpoints/lex_dest_exact before safety42-46. "
            "Final candidate remains 20-row exact-source lex_dest_exact; do not select new rows from validation.\n")
print("\nFRESH LANGUAGE REQUEST:")
print(json.dumps(fresh_req,separators=(",",":")))
```
