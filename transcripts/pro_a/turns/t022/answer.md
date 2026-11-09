The probe tool does not allow modified inputs, so I will not use those counterfactual files or try to bypass that restriction. Instead, I will run listed practice episodes that share the same scene and initial-state index across perturbations. Their saved initial images will show which pairs provide a valid instruction contrast and which also change visual content.
```python
# Only listed practice episodes from here; no modified-input probes.
grouped=collections.defaultdict(dict)
for e in dev_pool:
    grouped[(e["suite"],e["task"],e["init_idx"])][e["ptype"]]=e
complete={k:v for k,v in grouped.items() if len(v)==5}
print("Complete five-way groups by suite:",dict(collections.Counter(k[0] for k in complete)))
matched_groups=[]
for suite in sorted({e["suite"] for e in episodes}):
    keys=sorted([k for k in complete if k[0]==suite],
                key=lambda k:(hashlib.sha256(("/".join(map(str,k))+"/diag").encode()).hexdigest()))
    seen_tasks=set()
    for k in keys:
        if k[1] in seen_tasks: continue
        matched_groups.append(k);seen_tasks.add(k[1])
        if len(seen_tasks)==4: break
matched_eps=[complete[k][typ] for k in matched_groups for typ in
             ["task","semantic","position","environment","object"]]
print("Matched groups:",len(matched_groups),"episodes:",len(matched_eps))
print("Groups:",[(k[0],k[1],k[2]) for k in matched_groups])
matched_request=[{"id":e["id"],"seed":7} for e in matched_eps]
print("Request:",json.dumps(matched_request))
(W/"matched_development.json").write_text(json.dumps(matched_request))
with (W/"notes.md").open("a") as f:
    f.write("Counterfactual probe REFUSED (did not count): tool permits only exact rollout inputs. "
            "No workaround attempted. Switching to listed practice episodes matched on scene/init_idx across all five ptypes. "
            "Will check actual image equality before interpreting instruction contrasts.\n")
```
