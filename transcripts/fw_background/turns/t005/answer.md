Background variants use both `_table_` and `_tb_` suffixes. I will pair one harder background variant per task with a low-difficulty lighting variant at the same seed, and record residual features throughout both experts. This comparison can reveal where texture-related changes enter the action pathway, while avoiding comparisons between unrelated tasks.

```python
def task_stem(v):
    s = v["name"]
    if v["category"] == "Background Textures":
        return re.sub(r"_(?:table|tb)_\d+$", "", s)
    if v["category"] == "Light Conditions":
        return re.sub(r"_light_\d+$", "", s)
    if v["category"] == "Objects Layout":
        return re.split(r"_(?:add|remove|change|layout)_\d+", s)[0]
    return re.split(r"_(?:language|view|noise|initstate)_", s)[0]
for v in variants:
    v["_task"] = task_stem(v)
task_groups = defaultdict(list)
for v in variants:
    if v["category"] == "Background Textures" and v["_split"] == "diagnosis":
        task_groups[(v["suite"], v["_task"])].append(v)
diag_bg, diag_controls = [], []
for key, vs in sorted(task_groups.items()):
    vs.sort(key=lambda v: (-(v["difficulty"] or 0), v["id"]))
    cs = [v for v in bycat["Light Conditions"]
          if (v["suite"], v["_task"]) == key and v["_split"] == "diagnosis"]
    cs.sort(key=lambda v: (v["difficulty"] if v["difficulty"] is not None else 0, v["id"]))
    if cs:
        diag_bg.append(vs[0])
        diag_controls.append(cs[0])
diag_ep = [{"suite": v["suite"], "id": v["id"], "seed": 17}
           for v in diag_bg + diag_controls]
(W / "diagnostic_episodes.json").write_text(json.dumps(diag_ep))
print("Paired tasks:", len(diag_bg))
print("BG difficulty:", dict(Counter(v["difficulty"] for v in diag_bg)))
print("Control difficulty:", dict(Counter(v["difficulty"] for v in diag_controls)))
print(json.dumps(diag_ep, separators=(",", ":")))
```
