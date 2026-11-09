Before applying the projector, I’ll check input order and readout calibration across passes. Even a small mismatch could make the projection remove ordinary language variation instead of tag effects.
```python
print((W/"weighted_readout.py").read_text()[:5800])
base5=json.loads((W/"probe_5/results.json").read_text())
sid=json.loads((W/"screen_ids.json").read_text())
expected=[base5[i]["input"] for i in sid]+[base5[120+i]["input"] for i in sid]+[base5[i]["input"] for i in range(240,272)]
for k in [18,19,20,21,26,27,28]:
    rr=json.loads((W/f"probe_{k}/results.json").read_text())
    mis=[i for i,(a,b) in enumerate(zip(rr,expected)) if a["input"]!=b]
    print("pass",k,"n",len(rr),"order mismatch",mis[:30])
    if mis:
        for i in mis[:2]:
            print(i,Path(rr[i]["input"]).parent.name,Path(expected[i]).parent.name)
print("weighted and masked query definitions")
print(spu["queries"])
print(spec["queries"])
print("detector dot directions", (spec["directions"].T @ spec["detector_weight"]).tolist())
print("prior_candidate files",len(list((W/"prior_candidate").glob("*.npy"))))
print("last prior notes", (W/"prior_notes.md").read_text()[-1700:])
```
