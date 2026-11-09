I will compare first-chunk motion and gripper changes and check whether later heads redistribute attention. A useful repair should change inappropriate approach decisions without simply making the gripper inert.

```python
pr7=json.loads((W/"probe_7/results.json").read_text())
probe7_summary=paired_probe_summary(pr7,{8:[2],14:[1,4],15:[2],17:[1,7]})
# Identify the most repetitive and successful examples for later fixed-observation tests.
allbase=valid1+valid4
print("baseline failure counts by level:")
for lev in ["L1","L2","L3","L4"]:
    rr=[r for r in allbase if r["level"]==lev]
    print(lev,len(rr),sum(r["success"] for r in rr),
          "loops",sum(not r["success"] and r["grip_flips"]>=6 for r in rr))
```
