The reverse dose also failed: 57/96 became 54/96, and loop failures increased. I will stop proposing new edits. The best use of the remaining passes is independent replication of the two earlier frozen candidates that had the strongest diagnostic gains, with no retuning. Neither will be submitted unless it clears fresh-scene validation and a preservation check.
```python
for n in list(edited()):
    drop_edit(n)
parts=json.loads((W/"partitions.json").read_text())
print("Partition metadata keys:",list(parts))
for k,v in parts.items():
    if k!="partition":
        print(k,str(v)[:600])

for folder in ["best_H15","candidate_I065"]:
    print(folder,[(p.name,p.stat().st_size) for p in (W/folder).iterdir()][:8])

def source_group(s):
    scene=s["scene_id"]
    scene=scene.removeprefix("l4_injected__")
    return re.sub(r"__step\d+$","",scene)

vb=[s for s in scenes if parts["partition"][str(s["id"])]=="validation_B"
    and s["stage"] in ["mid-fail","post-fail"]]
selected=[]
used_groups=set()
for level in ["L1","L2","L3","L4"]:
    for suite in ["libero_10","libero_goal","libero_object","libero_spatial"]:
        cand=[s for s in vb if s["level"]==level and s["suite"]==suite]
        cand.sort(key=lambda s:hashlib.sha256(("confirm_B_2026:"+str(s["id"])).encode()).hexdigest())
        chosen=[]
        for s in cand:
            if source_group(s) not in used_groups:
                chosen.append(s)
                used_groups.add(source_group(s))
            if len(chosen)==4: break
        selected.extend(chosen)
        print(level,suite,"selected",len(chosen))
valB_episodes=[{"id":s["id"],"grip":g,"seed":43} for s in selected for g in ["open","close"]]
(W/"validation_B_episodes.json").write_text(json.dumps(valB_episodes))
print("B scenes",len(selected),"episodes",len(valB_episodes),
      "IDs",[s["id"] for s in selected])

with (W/"notes.md").open("a") as f:
    f.write("\n## N reverse dose rejected, pass44\n"
            "57/96->54/96,-3.1+/-2.8pp,2 rescues/5 regressions,loops35->38. Reset.\n"
            "\n## Final replication plan (no new tuning)\n"
            "Use untouched validationB to replicate frozen H15 and I065, the two strongest "
            "earlier diagnostic candidates. No parameter changes based on B. "
            "Retention requires positive B and pooled validationA+B gain>2 paired SE, "
            "then another fresh-scene confirmation and a pre-fail preservation check. "
            "If neither qualifies, submit original weights, not a speculative modification.\n")
```
