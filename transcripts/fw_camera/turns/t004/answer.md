Some practice entries have no difficulty label. I will exclude those from the matched severity comparison rather than assume they are mild, and keep the task/seed matching unchanged.```python
print("Missing difficulty by category:",
      dict(collections.Counter(x["category"] for x in practice if x["difficulty"] is None)))
diag_variants, diag_groups = [], []
rng = np.random.default_rng(1701)
for suite in sorted(set(x["suite"] for x in practice)):
    eligible = []
    for (s, task), xs in by_task.items():
        if s != suite:
            continue
        lo = [x for x in xs if x["category"] == "Camera Viewpoints" and x["difficulty"] in (1, 2)]
        hi = [x for x in xs if x["category"] == "Camera Viewpoints" and x["difficulty"] in (4, 5)]
        ctrl = [x for x in xs if x["category"] == "Light Conditions" and x["difficulty"] in (1, 2)]
        if lo and hi and ctrl:
            eligible.append((task, lo, hi, ctrl))
    eligible.sort(key=lambda z: z[0])
    print(suite, "eligible tasks:", len(eligible))
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
print("Episodes:", json.dumps(diag_episodes, separators=(",", ":")))
```
