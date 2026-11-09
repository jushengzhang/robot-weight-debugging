The first visual edit produced three wins and no losses, but the sample is too small to call it a reliable improvement, and the aggregate loop count did not decrease. I’ll test full removal of the same two directions as a dose-response experiment; I won’t change the basis or add more directions based on these outcomes.

```python
r7=read_roll(7)
print("half-dose results:",paired_summary(r7))
pairs7=collections.defaultdict(dict)
for r in r7: pairs7[r["id"]][r["model"]]=r
print("changed outcomes:",[(eid,v["original"]["ptype"],
      v["original"]["success"],v["edited"]["success"],
      v["original"]["grip_flips"],v["edited"]["grip_flips"])
      for eid,v in pairs7.items() if v["original"]["success"]!=v["edited"]["success"]])
for arm in ("original","edited"):
    es=[r for r in r7 if r["model"]==arm and r["ptype"]=="environment"]
    print(arm,"env mean flips",round(np.mean([r["grip_flips"] for r in es]),2),
          "mean path",round(np.mean([r["eef_path_m"] for r in es]),3))
print("full-dose:",environment_edit(beta=1.0,rank=2))
snapshot("candidate_environment_r2_b1")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 7: environment half-dose diagnostic\n"
            "44/75 ->47/75 (+4.0 +/-2.3 pp paired SE), 3 wins/0 losses; "
            "env7->9, pos4->5, task4 unchanged, object14 and semantic15 unchanged. "
            "Aggregate loop failures did not fall (14->16), so no loop-specific claim yet. "
            "Next dose is full removal of identical two directions, no basis re-estimation.\n")
```
