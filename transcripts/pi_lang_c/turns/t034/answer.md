The 80 goal pairs now complete without runtime errors: success rises from 49/80 to 69/80, with 20 recoveries and no regressions. I will save this candidate, compare it against the earlier noun-only edit on common episodes, and prepare the first completely disjoint confirmation set.
```python
print("C broad:",paired_summary(14))
print("C goal:",paired_summary(15))
r14=read_roll(14)
for r in r14:
    if r.get("termination")=="error":
        print("Error record:",json.dumps(r,ensure_ascii=True)[:1500])
raw14=[json.loads(x) for x in (W/"fwd/roll_14/results.jsonl").read_text().splitlines() if x.strip()]
for r in raw14:
    if r.get("failure") and (str(r["failure"]).startswith("error") or r.get("termination")=="error"):
        if "OutOfMemory" not in str(r):
            print("Raw error:",json.dumps(r,ensure_ascii=True)[:1800])
def compare_edits(ka,kb):
    aa={(r["suite"],r["id"],r["seed"]):r for r in read_roll(ka) if r["model"]=="edited" and r.get("termination") in ("success","max_steps")}
    bb={(r["suite"],r["id"],r["seed"]):r for r in read_roll(kb) if r["model"]=="edited" and r.get("termination") in ("success","max_steps")}
    common=sorted(set(aa)&set(bb))
    changes=[]
    for key in common:
        if aa[key]["success"]!=bb[key]["success"]:
            changes.append((key,aa[key]["success"],bb[key]["success"],bb[key]["instruction"]))
    print("Edit comparison",ka,kb,"common",len(common),"changes",changes)
compare_edits(10,14); compare_edits(12,15)
torch.save({"rows":rows_C,"strengths":strengths_C,"aliases":{**candidate_aliases,**verb_aliases}},W/"best_C_recipe.pt")
if not (W/"best_C").exists(): shutil.copytree(W/"edited",W/"best_C")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate C (passes 14-15)\n"
            "Broad diagnostic "+str(paired_summary(14))+". Goal diagnostic: 49/80->69/80, "
            "+25.0 pp SE4.9, 20 recoveries, 0 regressions; no OOMs. "
            "Saved best_C/. Candidate now frozen for disjoint confirmation.\n")
print("Confirmation 1 suite/difficulty counts:",
      sorted(collections.Counter((v["suite"],v["difficulty"]) for v in confirm_sets[0]).items()))
print("CONFIRM_1_JSON:",json.dumps(json.loads((W/"confirm_1_episodes.json").read_text()),separators=(",",":")))
```
