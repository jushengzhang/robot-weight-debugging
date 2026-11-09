The practice list contains all five perturbation types but no entries labeled “original.” I will use matched scene/init-state groups across perturbations for diagnosis, with successful object/semantic cases as controls, and check whether the rollout tool provides a separate original-task control evaluation.
```python
by_scene = collections.defaultdict(lambda: collections.defaultdict(dict))
for e in eps:
    by_scene[(e["suite"], e["task"])][e["init_idx"]][e["ptype"]] = e
ptypes = ["object", "position", "semantic", "task", "environment"]
diag_eps = []
chosen_scenes = []
for suite in sorted({e["suite"] for e in eps}):
    candidates = []
    for (s,t), states in by_scene.items():
        if s != suite:
            continue
        full = [i for i,x in states.items() if all(p in x for p in ptypes)]
        if full:
            candidates.append((t, min(full)))
    for t, i in sorted(candidates)[:4]:
        chosen_scenes.append((suite, t, i))
        diag_eps += [by_scene[(suite,t)][i][p] for p in ptypes]
print("matched scenes:", len(chosen_scenes), "episodes:", len(diag_eps))
for s,t,i in chosen_scenes:
    group=by_scene[(s,t)][i]
    print(s, i, t)
    print("  changed instruction:", group["task"]["instruction"])
diag_request = [{"id":e["id"],"seed":17} for e in diag_eps]
(W / "diagnostic_episodes.json").write_text(json.dumps(diag_request))
print("diagnostic requests:", json.dumps(diag_request))
print("work entries:", sorted(p.name for p in W.iterdir())[:25])
```
