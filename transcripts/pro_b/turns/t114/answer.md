The positional-row edit did not improve overall success, so I will discard it. With five passes left, I will stop proposing new edits and run a frozen, higher-powered confirmation of the earlier conservative environment-projector edit. It had small positive overall results in two samples, but neither was conclusive. I will test 640 untouched confirmation episodes and retain it only if the aggregate gain exceeds twice its paired standard error, with no object/semantic regression and no environment loss.
```python
for n in list(edited()):drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes34-35: extreme-position edit rejected\n"
            "Half excess on5outlier positional rows changes late grounding attention TV .08-.18, early fixed-wrist heads mostly unchanged. "
            "XYZ probe RMS .009-.025, much smaller rotation/gripper changes. "
            "Rollout66/128->65/128 (-0.8 +/-1.8pp),position7->8,environment16->14,task unchanged; loops28->39. Reverted.\n")
# Restore the earlier frozen rank-2 half-removal, not a newly estimated edit.
for p in (W/"candidate_environment_r2_b05").glob("*.npy"):
    n=p.name[:-4]
    save_edit(n,torch.from_numpy(np.array(np.load(p))).float())
print("Frozen candidate",edited())
for n in edited():
    x=load(n);x0=original_clone(n)
    print(n.split("model.")[-1],"relative delta",round(float((x-x0).norm()/x0.norm()),6))
# Use only untouched confirmation episodes, selected without outcome information.
used=set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        r=json.loads(line);used.add(r["id"])
confirm=set(manifest["confirmation"])
print("Previously used confirmation IDs",len(confirm&used))
rng=np.random.default_rng(362025)
batches=[[] for _ in range(5)]
chosen_conf=[]
ptypes=["environment","object","position","semantic","task"]
suites=sorted(set(r["suite"] for r in pr))
s=0
for suite in suites:
    for pt in ptypes:
        pool=sorted([r for r in pr if r["id"] in confirm and r["id"] not in used
                     and r["suite"]==suite and r["ptype"]==pt],key=lambda r:r["id"])
        assert len(pool)>=32,(suite,pt,len(pool))
        picks=[pool[int(j)] for j in rng.permutation(len(pool))[:32]]
        for j,r in enumerate(picks):
            batches[(j+2*s)%5].append(r["id"]);chosen_conf.append(r["id"])
        s+=1
assert len(set(chosen_conf))==640 and [len(b) for b in batches]==[128]*5
decision={"candidate":"environment_projector_rank2_beta0.5",
          "batches":batches,"seed":43,"planned_passes":[36,37,38,39,40],
          "criterion":"overall delta > 2*max(paired_SE, grouped_cluster_SE); "
                      "object+semantic delta >= 0; each of object,semantic delta >= -2pp; environment delta >= 0",
          "no_selection_or_edit_change_between_batches":True}
(W/"final_confirmation_plan.json").write_text(json.dumps(decision,indent=2))
for k,ids in enumerate(batches):
    req={"type":"rollout","model":"both","episodes":[{"id":i,"seed":43} for i in ids]}
    (W/f"final_confirmation_request_{k+1}.json").write_text(json.dumps(req))
print("Batch sizes",[len(b) for b in batches],
      "type totals",dict(collections.Counter(byid[i]["ptype"] for i in chosen_conf)))
print("Unique suite/task/init clusters",len(set((byid[i]["suite"],byid[i]["task"],byid[i]["init_idx"])
                                               for i in chosen_conf)))
print("Batch 1 request:")
print((W/"final_confirmation_request_1.json").read_text())
with (W/"notes.md").open("a") as f:
    f.write("\n## Final frozen high-power confirmation plan (passes36-40)\n"
            "No new candidate families. Revisit only frozen environment-projector rank2 half-removal from pass7: "
            "initial+3/75, first disjoint sample+2/128 overall, but environment-specific result inconclusive/negative. "
            "Not currently validated; original remains default final choice. "
            "Test640 untouched confirmation episodes,128perptype,32per(suite,ptype),5fixed batches128,paired seed43. "
            "No parameter/episode changes after seeing results. Retain ONLY if overall gain >2*max(pairedSE,clusterSE), "
            "object+semantic gain>=0, neither object nor semantic drop>2pp, and environment gain>=0. "
            "Otherwise submit no edits. Plan and complete IDs in final_confirmation_plan.json. "
            "Unperturbed episodes are unavailable in practice list; cannot directly measure original-task regression.\n")
```
