Several unrelated small edits rescued and regressed the same episodes, which could reflect numerical trajectory sensitivity rather than the proposed mechanisms. Before attributing further effects to a head, I will compare repeated original action traces and run a no-edit paired control. A valid causal test needs the two execution paths to agree when the weights agree.
```python
rolls={k:read_roll(k) for k in [6,8,10,12,15]}
base6={key:d["original"] for key,d in pair_records(rolls[6]).items()}
for k,rr in rolls.items():
    dd=pair_records(rr)
    adiffs=[]; fchanges=[]; outcomechanges=[]
    for key,bo in base6.items():
        ro=dd[key]["original"]
        a=np.load(bo["actions_npy"]); b=np.load(ro["actions_npy"])
        adiffs.append(float(np.max(np.abs(a-b))) if a.shape==b.shape else float("inf"))
        if bo["grip_flips"]!=ro["grip_flips"]:fchanges.append((key,bo["grip_flips"],ro["grip_flips"]))
        if bo["success"]!=ro["success"]:outcomechanges.append(key)
    print("original repeat",k,"max action difference",max(adiffs),
          "nonidentical",sum(x!=0 for x in adiffs),"flipcount differences",fchanges[:5],
          "outcome differences",outcomechanges)
# Preserve this prospective ablation specification, then run a strict null-edit control.
reset_edits()
print("sham edited tensor count",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Execution-path control\nRepeated small edits often change the same four outcomes "
            "(1102 open and1300 close rescued;935 open and126 open regressed). "
            "Before further causal claims, check paired original/edited with zero saved edits.\n")
```
