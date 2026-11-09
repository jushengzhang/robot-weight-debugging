Gain 4 reduced both task and semantic success, so I will remove the value edit. The next diagnosis will test a different hypothesis: scene-specific MLP outputs may dominate the residual stream even when attention carries instruction information. I will first compare those outputs across matched scene pairs.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass16: reject E gain4\n71->68/128, -2.3pp SE2.3; task8->6, semantic27->25, position4->5, env/object unchanged. Old-goal failures11->12, loops30->38. No supporting dose-response; all E edits removed.\n")
print("cleared",edited())
print("probe extras",[(k,az[k].shape) for k in az.files[65:]])
print("probe input paths")
for i,r in enumerate(ix14):
    print(i,r["input"].replace(str(W),"WORK"))
r1=[json.loads(s) for s in (W/"roll_1.jsonl").read_text().splitlines()]
print("matching initial diagnostic tasks")
for r in r1:
    if r["ptype"]=="task":
        print(r["id"],r["suite"],r["task"][:60],r["success"],r["orig_goal_reached"],r["instruction"])
```
