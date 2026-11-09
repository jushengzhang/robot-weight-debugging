The failures are concentrated in descriptive substitutions such as “fermented grape beverage,” “storage unit,” and “cooking appliance,” while many long instructions retaining the literal object names succeed. I will inspect these substitutions and replay metadata more compactly, then establish same-task controls.
```python
for r in r1[45:]:
    print(r["id"], int(r["success"]), r["instruction"])
print("input file", str(probe_example).replace(str(W),"WORK"))
print("input structure", json.dumps(structure(inp), ensure_ascii=True)[:2200])
print("controls", [(v["id"], v["name"].split("_language_")[0],v["difficulty"]) for v in easy_goal])
print("goal tasks", tasks_goal)
print("activation example", [(k,np.array(v).shape) for k,v in np.load(r1[0]["activations"]).items()])
with (W/"notes.md").open("a") as f:
    f.write("\nPass 1: original goal language d4-5, 31/64 = 48.4%; d4 13/17, d5 18/47. "
            "Descriptive object substitutions dominate failures; long context with canonical names often works.\n")
```
