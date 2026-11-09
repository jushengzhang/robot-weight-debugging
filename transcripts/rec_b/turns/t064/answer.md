The stronger dose is the first clearly promising tuning result: five rescues and one regression, with loop failures falling from 21 to 16. I will freeze this edit and select an untouched, source-demonstration-separated validation set before examining any more outcomes.
```python
import hashlib, shutil
bestdir=W/"best_H15"
if bestdir.exists():shutil.rmtree(bestdir)
shutil.copytree(W/"edited",bestdir)
pp=pairs(25)
print("Changed episodes:")
for o,e in pp:
    if o["success"]!=e["success"]:
        print(o["id"],o["grip"],o["level"],o["instruction"],
              "success",o["success"],"->",e["success"],
              "steps",o["steps"],"->",e["steps"],
              "flips",o["grip_flips"],"->",e["grip_flips"],
              "path",o["eef_path_m"],"->",e["eef_path_m"],
              "net",o["eef_net_m"],"->",e["eef_net_m"])
solved=[(o,e) for o,e in pp if o["success"]]
print("Baseline-solved preserved",sum(e["success"] for o,e in solved),"/",len(solved))
print("Partitions by label",Counter(parts["partition"].values()))
def source_group(s):
    v=s["scene_id"]
    v=re.sub(r"^l4_injected__","",v)
    return re.sub(r"__step\d+.*$","",v)
print("Grouping example",[(r["id"],source_group(r)) for r in practice[:2]])
def fixedrank(s,label):
    return hashlib.sha256((label+"|"+source_group(s)+"|"+str(s["id"])).encode()).hexdigest()
# Build validation set without looking at any outcomes.
pool=[r for r in practice if parts["partition"][str(r["id"])]=="validation_A"
      and r["stage"]!="pre-fail"]
va=[];usedgroups=set()
for lev in ["L1","L2","L3","L4"]:
    for suite in sorted(set(r["suite"] for r in practice)):
        cand=sorted([r for r in pool if r["level"]==lev and r["suite"]==suite],
                    key=lambda s:fixedrank(s,"frozen-validation-A"))
        got=0
        for r in cand:
            if source_group(r) in usedgroups:continue
            va.append(r);usedgroups.add(source_group(r));got+=1
            if got==4:break
    # Fill to 16 unique sources per level if a suite lacks enough.
    need=16-sum(r["level"]==lev for r in va)
    for r in sorted([r for r in pool if r["level"]==lev],key=lambda s:fixedrank(s,"frozen-validation-A")):
        if need<=0:break
        if source_group(r) in usedgroups:continue
        va.append(r);usedgroups.add(source_group(r));need-=1
vaeps=[{"id":r["id"],"grip":grip,"seed":29} for r in va for grip in ["open","close"]]
(W/"validation_A_episodes.json").write_text(json.dumps(vaeps,indent=2))
print("Validation A scenes",len(va),"cells",dict(Counter((r["level"],r["suite"]) for r in va)))
print("Validation request episodes:",json.dumps(vaeps))
with (W/"notes.md").open("a") as f:
    f.write("\n## H amplification1.5 promising, pass25\n"
            "Q2h0 grip-specific gain2.5x:26/48->30/48,+8.3+/-5.0pp,5 rescues/1 regression,loops21->16. "
            "Dose series alpha=-.5:23/48,0:26/48,+.5:27/48,+1.5:30/48. "
            "Current edit saved in best_H15/. Frozen for independent validationA:64 scenes,128 paired episodes,"
            "seed29,source-demo-separated,balanced level/suite,no pre-fail. No validation results used for selection yet.\n")
```
