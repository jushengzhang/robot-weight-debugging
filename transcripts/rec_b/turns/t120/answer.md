The routing edit preserved the pre-fail aggregate at 86/96, but exchanged three successes for three failures and increased mean completion time. All these scenes are from `libero_object`, so this cannot certify ordinary performance across LIBERO. I will run the same preservation panel for H15, then remove both candidates.
```python
print("I065 pre-fail preservation:",pooled_stats([47]))
restore_snapshot("best_H15")
print("H15 restored unchanged:",edited())

r47=read_jsonl(W/"roll_47.jsonl")
controls=[r for r in r47 if r["model"]=="original" and r["success"]
          and r["seed"]==7 and r["grip"]=="open" and r["steps"]>50
          and r["level"]=="L2"]
controls=sorted(controls,key=lambda r:r["id"])
print("Available pre-fail controls:",
      [(r["id"],r["steps"],r["instruction"]) for r in controls[:5]])
with (W/"notes.md").open("a") as f:
    f.write("\n## I065 pre-fail preservation, pass47\n"
            "16 reserved scenes x2 grips x3 seeds=96 episodes, all libero_object. "
            "86/96->86/96,3 rescues/3 regressions,loops10->10; "
            "mean successful steps129.2->136.6. Not an ordinary LIBERO certification. "
            "H15 restored unchanged for same preservation audit.\n")
```
