The weights expose separate video keys/values and action queries/output projections in every shared-attention layer, which gives us several targeted intervention points. I will build an initial diagnostic panel with mild and severe camera changes plus low-difficulty lighting controls on the same tasks and seeds; this comparison should help distinguish viewpoint sensitivity from general task difficulty.```python
def task_name(x):
    n = x["name"]
    for marker in ("_language_", "_view_", "_light_", "_table_", "_floor_",
                   "_wall_", "_add_", "_remove_", "_noise_"):
        if marker in n:
            n = n.split(marker)[0]
    return n

by_task = collections.defaultdict(list)
for x in discovery:
    by_task[(x["suite"], task_name(x))].append(x)
diag_variants = []
diag_groups = []
rng = np.random.default_rng(1701)
for suite in sorted(set(x["suite"] for x in practice)):
    eligible = []
    for (s, task), xs in by_task.items():
        if s != suite:
            continue
        lo = [x for x in xs if x["category"] == "Camera Viewpoints" and x["difficulty"] <= 2]
        hi = [x for x in xs if x["category"] == "Camera Viewpoints" and x["difficulty"] >= 4]
        ctrl = [x for x in xs if x["category"] == "Light Conditions" and x["difficulty"] <= 2]
        if lo and hi and ctrl:
            eligible.append((task, lo, hi, ctrl))
    eligible.sort(key=lambda z: z[0])
    for j in rng.permutation(len(eligible))[:8]:
        task, lo, hi, ctrl = eligible[j]
        chosen = [lo[int(rng.integers(len(lo)))],
                  hi[int(rng.integers(len(hi)))],
                  min(ctrl, key=lambda x: (x["difficulty"], x["id"]))]
        seed = 1100 + len(diag_groups)
        diag_groups.append({"suite": suite, "task": task, "seed": seed,
                            "ids": [x["id"] for x in chosen]})
        diag_variants.extend([{**x, "seed": seed} for x in chosen])
diag_episodes = [{k: x[k] for k in ("suite", "id", "seed")} for x in diag_variants]
(W / "diagnostic_panel.json").write_text(json.dumps(diag_variants, indent=2))
(W / "diagnostic_groups.json").write_text(json.dumps(diag_groups, indent=2))
print("Diagnostic counts:", dict(collections.Counter(x["category"] for x in diag_variants)))
print("Suites:", dict(collections.Counter(x["suite"] for x in diag_variants)))
print("Episodes:", json.dumps(diag_episodes, separators=(",", ":")))
```
