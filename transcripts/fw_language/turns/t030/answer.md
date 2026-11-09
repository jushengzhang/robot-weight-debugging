The saved inputs are accessible through per-episode symlinks, and some lighting controls have an unspecified difficulty. I will treat those separately rather than assume a numeric level, inspect their names, and select the least-perturbed same-task controls.
```python
inputs1=[p for r in roll1 for p in sorted(Path(r["frames_dir"]).glob("*_inputs.pt"))]
inp1=torch.load(inputs1[0],map_location="cpu",weights_only=False)
print("Saved inputs:",len(inputs1))
for k,v in inp1.items():
    print(k,json.dumps(small_desc(v),ensure_ascii=True)[:1200])
print("\nUnspecified-difficulty controls (examples):")
null_lights=[v for v in variants if v["category"]=="Light Conditions" and v["difficulty"] is None]
print("Count:",len(null_lights))
print(null_lights[:2])
control_eps=[]
control_map={}
for r in roll1:
    task=r["name"].split("_language_")[0]
    choices=by_task_cat[(r["suite"],task,"Light Conditions")]
    c=min(choices,key=lambda v:(v["difficulty"] if v["difficulty"] is not None else 0,v["id"]))
    key=(c["suite"],c["id"],r["seed"])
    if key not in control_map:
        control_map[key]=c
        control_eps.append({"suite":c["suite"],"id":c["id"],"seed":r["seed"]})
print("\nSelected controls:")
for c in control_map.values():
    print(c["suite"],c["id"],"difficulty",c["difficulty"],c["name"][-65:])
print("Request episode list:",json.dumps(control_eps,separators=(",",":")))
(W/"diagnostic_control_episodes.json").write_text(json.dumps(control_eps))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1 baseline (backend available)\n"
            "16 original language episodes: 12 successes, 4 failures. "
            "Failures: spatial1062 (noun circumlocutions), goal1188 "
            "(wine container/cupboard), goal1246 (implicit opening prerequisite), "
            "long1208 (coffee conversational phrasing). Descriptive only. "
            "Replay inputs reside in per-episode symlink directories. "
            "No edits. 1/40 passes used.\n")
```
