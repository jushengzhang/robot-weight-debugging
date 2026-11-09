Some practice variants have no difficulty label. I will handle those explicitly and use the task-name prefixes to find background variants and near-standard lighting controls of the same tasks.

```python
for cat in ["Light Conditions", "Sensor Noise", "Objects Layout", "Robot Initial States"]:
    print(cat, "examples:", bycat[cat][:2])
    print("difficulty:", dict(Counter(v["difficulty"] for v in bycat[cat])))
def split_of(v):
    h = hashlib.sha256(f"practice-split-v1:{v['suite']}:{v['id']}".encode()).digest()
    return "diagnosis" if int.from_bytes(h[:4], "little") % 3 == 0 else "validation"
for v in variants:
    v["_split"] = split_of(v)
print("Split counts:", dict(Counter((v["category"], v["_split"]) for v in variants)))
(W / "practice_split.json").write_text(json.dumps(
    [{"suite": v["suite"], "id": v["id"], "split": v["_split"]} for v in variants]))

# Only suffix parsing of the permitted practice catalogue.
markers = {
    "Background Textures": "_table_",
    "Light Conditions": "_light_",
    "Camera Viewpoints": "_view_",
    "Language Instructions": "_language_",
    "Robot Initial States": "_view_",
    "Sensor Noise": "_noise_",
    "Objects Layout": "_layout_",
}
def task_stem(v):
    s = v["name"]
    if v["category"] in ("Background Textures", "Light Conditions", "Camera Viewpoints", "Language Instructions"):
        return s.split(markers[v["category"]])[0]
    return re.split(r"_(?:view|noise|layout|initstate|sensor|object_layout)_", s)[0]
for v in variants:
    v["_task"] = task_stem(v)
bgdiag = [v for v in bycat["Background Textures"] if v["_split"] == "diagnosis"]
rng = random.Random(2401)
rng.shuffle(bgdiag)
# Two diagnosis variants per task where possible; balanced suite coverage.
task_groups = defaultdict(list)
for v in bgdiag:
    task_groups[(v["suite"], v["_task"])].append(v)
diag_bg = []
for key in sorted(task_groups):
    diag_bg.extend(task_groups[key][:2])
diag_bg = diag_bg[:80]
diag_controls = []
seen_ctrl = set()
for v in diag_bg:
    cs = [c for c in bycat["Light Conditions"]
          if c["suite"] == v["suite"] and c["_task"] == v["_task"]
          and c["_split"] == "diagnosis" and c["difficulty"] in (None, 1, 2)]
    cs.sort(key=lambda c: (c["difficulty"] if c["difficulty"] is not None else 0, c["id"]))
    if cs and (cs[0]["suite"], cs[0]["id"]) not in seen_ctrl:
        diag_controls.append(cs[0])
        seen_ctrl.add((cs[0]["suite"], cs[0]["id"]))
diag_ep = [{"suite": v["suite"], "id": v["id"], "seed": 17}
           for v in diag_bg + diag_controls]
(W / "diagnostic_episodes.json").write_text(json.dumps(diag_ep))
print("Diagnostic BG and controls:", len(diag_bg), len(diag_controls))
print("First diagnostic episodes:", diag_ep[:12])
print("BG suites:", dict(Counter(v["suite"] for v in diag_bg)))
print("Control IDs by suite:", {s:[v["id"] for v in diag_controls if v["suite"]==s] for s in sorted(set(v["suite"] for v in variants))})
```
