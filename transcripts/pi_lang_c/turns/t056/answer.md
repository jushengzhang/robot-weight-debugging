The third reserved batch also improves by +4.7 points. I will finish with the last 76 reserved variants and rerun every zero-step error pair from the previous two batches, so the pooled result is not driven by startup failures.
```python
retry_keys=set()
for k in [20,21,22]:
    for key,p in pair_roll(k).items():
        if any(r.get("termination") not in ("success","max_steps") for r in p.values()):
            retry_keys.add(key)
            print("Retry",k,key,[(m,r["termination"],r["steps"]) for m,r in p.items()])
last_eps=json.loads((W/"confirm_5_episodes.json").read_text())
last_eps += [{"suite":s,"id":i,"seed":seed} for s,i,seed in sorted(retry_keys)]
assert len(last_eps)<=128
print("Last batch + retries",len(last_eps))
print("EPISODES",json.dumps(last_eps,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\nF confirmation batch3/pass22: 107->113 /128, 8 recoveries/2 regressions, "
            "+4.69 pp SE2.45. Still no weight changes. Last76 reserved variants will run together "
            "with all zero-step error pairs from passes21/22, replacing those startup-error records in pooled analysis.\n")
```
