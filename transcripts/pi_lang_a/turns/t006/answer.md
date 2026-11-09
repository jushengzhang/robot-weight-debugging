The failures are concentrated in instructions that remove the familiar object names, but a few contextualized drawer instructions also fail. I will add easy same-task references, several hard stove instructions, and a modest diagnostic sample from the other suites. The remaining practice variants will be reserved for later confirmation.

```python
rng = np.random.default_rng(2026)
seen1 = {(r["suite"], r["id"]) for r in r1}
diag2 = []
for task in goal_tasks:
    choices = [v for v in goal_lang if task_of(v) == task and v["difficulty"] in [1,2]
               and (v["suite"],v["id"]) not in seen1]
    choices.sort(key=lambda v: (v["difficulty"],v["id"]))
    if choices:
        diag2.append(choices[0])
stove_hard = [v for v in goal_lang if task_of(v) == "turn_on_the_stove"
              and v["difficulty"] == 5 and (v["suite"],v["id"]) not in seen1]
diag2.extend(stove_hard[:4])
for suite in ["libero_spatial","libero_object","libero_10"]:
    pool = [v for v in variants if v["suite"] == suite and v["category"] == "Language Instructions"
            and v["difficulty"] in [4,5]]
    diag2.extend([pool[i] for i in rng.choice(len(pool), 24, replace=False)])
diag2_eps = [{"suite":v["suite"],"id":v["id"],"seed":7} for v in diag2]
(W/"diagnostic_2_episodes.json").write_text(json.dumps(diag2_eps))
print("diagnostic episode count:",len(diag2_eps))
print(json.dumps(diag2_eps, separators=(",",":")))
diagnostic_keys = seen1 | {(v["suite"],v["id"]) for v in diag2}
(W/"diagnostic_keys.json").write_text(json.dumps(sorted(diagnostic_keys)))
```
