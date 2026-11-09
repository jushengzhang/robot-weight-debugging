All six non-language categories have now been checked on 128 paired episodes each, with no negative success-rate change. I will next use saved observations to separate model effects from simulator variability and trace how the repaired language embeddings change the action expert’s task representation.
```python
safety={}
for k in range(24,30):
    ps=pair_roll(k)
    cat=next(iter(ps.values()))["original"]["category"]
    safety[cat]=paired_stats(ps)
print("SAFETY",json.dumps(safety,sort_keys=True))
print("DIAGNOSTIC PATH FILE")
oldpaths=json.loads((W/"diagnostic_probe_paths.json").read_text())
print(str(oldpaths)[:3800])
print("LITERAL GOAL CONTROLS")
for r in read_roll(2):
    print(r["suite"],r["id"],r["instruction"],r.get("frames_dir","").split("/")[-1])
print("PROBE RESULTS SCHEMA")
pr4=json.loads((W/"probe_4"/"results.json").read_text())
print(type(pr4).__name__,str(pr4)[:2400])
print("SAVED INPUT ELEMENT SCHEMA")
print({k:(type(v).__name__,getattr(v,"shape",None)) for k,v in inp["element"].items()})
with (W/"notes.md").open("a") as f:
    f.write("\n## Safety sweep completed, passes24-29\n"+json.dumps(safety,sort_keys=True)+
            "\nAll safety prompts have no edited token. Some simulated trajectories differ despite that; "
            "do not attribute the +2 lighting successes to the edit. Camera episode-mean activations differ "
            "in only3/128 trajectories, including vision, consistent with different observations. "
            "Will use identical-input probes for a proper invariance test. Final F remains frozen.\n")
```
