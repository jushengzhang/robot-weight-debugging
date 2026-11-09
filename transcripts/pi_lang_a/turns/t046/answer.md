The first independent batch gained 4.7 percentage points (paired SE 1.9), with six rescues and no regressions. I will save that result and prepare the remaining object and spatial validation batch without inspecting its instructions or changing the candidate.
```python
s17,_=paired_summary(records(17))
print("Pass17:",s17)
request18={"type":"rollout","model":"both","episodes":batch18}
(W/"request18.json").write_text(json.dumps(request18))
with (W/"notes.md").open("a") as f:
    f.write("\nPass17 independent stageA goal64+long64: orig101/128 edited107/128,+4.69pp(SE1.88),6rescues0regressions. Goal47->53/64; long54->54/64. No validation instructions inspected and no candidate changes.\n")
print(json.dumps(request18,separators=(",",":")))
```