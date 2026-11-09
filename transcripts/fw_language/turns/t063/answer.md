The video-only intervention produced no net gain: two rescues and two regressions, or 0.0 +/- 1.6 percentage points. I will test the action-only branch next and check whether the repeated original runs used the same episode identities, since their aggregate baselines differed slightly.
```python
r10=[json.loads(s) for s in (W/"roll_10.jsonl").read_text().splitlines()]
def rec_key(r): return (r["suite"],int(r["id"]),int(r["seed"]))
p3={rec_key(r):r for r in r3}
p10={rec_key(r):r for r in r10 if r["model"]=="original"}
common=set(p3)&set(p10)
flips=[k for k in common if p3[k]["success"]!=p10[k]["success"]]
print("Common episode identities:",len(common),"/",len(p3),len(p10))
print("Original rerun outcome changes:",len(flips),flips[:10])
print("Seeds:",sorted(set(r["seed"] for r in r10)))
print(set_head_output_edit(["action"],13,14,0.5,"L13H14_action_half"))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 10\n"
            "- Video-only L13 H14 half-gain: original 89/128, edited 89/128; 0.0 pp +/-1.6 SE; 2 rescues, 2 regressions. No benefit. Reverted before action-only test.\n"
            "- Original repeated baseline differs slightly from pass 3/7 despite same requested sample; use within-pass paired estimates rather than across-pass outcome subtraction.\n")
```
