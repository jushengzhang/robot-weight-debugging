Many failures replace task-defining nouns with descriptions, such as “rounded vessel” for bowl or “cooker” for stove. I will print a compact task-by-task comparison and inspect the saved tensors; this should reveal whether the action expert receives a weakened task signal or reads the wrong signal.

```python
def task_of(v):
    return v["name"].split("_language_")[0]
goal_tasks = sorted({task_of(v) for v in goal_lang})
for t in goal_tasks:
    print("\nTASK:", t)
    rr = [r for r in r1 if task_of(r) == t]
    for r in rr:
        if not r["success"] or r["difficulty"] <= 2:
            print(f'  {r["id"]} d{r["difficulty"]} {"OK" if r["success"] else "FAIL"}: {r["instruction"]}')
a1 = np.load(r1[0]["activations"])
print("\nactivation keys/shapes:", [(k, a1[k].shape) for k in a1.files][:12])
print("saved inputs:", [p.name for p in Path(r1[0]["frames_dir"]).glob("*")][:6])
print("turn-on-stove records:")
for r in r1:
    if "turn_on" in r["name"]:
        print(r["id"], r["success"], r["instruction"])
print("thread count:", torch.get_num_threads())
```
