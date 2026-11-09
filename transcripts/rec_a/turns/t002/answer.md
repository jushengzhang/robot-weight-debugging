I will use a level-balanced discovery set, with both gripper starts, to avoid letting the common L2 scenes dominate the diagnosis. The initial rollout will save actions, images, and attention-output summaries across the action expert; these will let me compare looping failures with successful recoveries before making any edits.
```python
scene_by_id = {s["id"]: s for s in scenes}
disc_set = set(discovery)
rng = np.random.default_rng(1729)
diag_ids = []
for level in ("L1", "L2", "L3", "L4"):
    pools = {}
    for suite in sorted({s["suite"] for s in scenes}):
        ids = [s["id"] for s in scenes if s["id"] in disc_set and s["level"] == level and s["suite"] == suite]
        rng.shuffle(ids)
        pools[suite] = ids
    picked = []
    while len(picked) < 16 and any(pools.values()):
        for suite, ids in pools.items():
            if ids and len(picked) < 16:
                picked.append(ids.pop())
    diag_ids.extend(picked)
diag_episodes = [{"id": i, "grip": g, "seed": 17} for i in diag_ids for g in ("open", "close")]
(W / "diagnosis_episodes.json").write_text(json.dumps(diag_episodes, indent=2))
req = {
    "type": "rollout", "model": "original", "episodes": diag_episodes,
    "record": {"expert_layers": list(range(18)), "vlm_layers": [0,3,6,9,12,15,17],
               "vision_layers": [0,13,26], "what": "attn_out"},
    "save_frames": True
}
print("Discovery composition:", dict(collections.Counter((scene_by_id[i]["level"],scene_by_id[i]["suite"]) for i in diag_ids)))
print(json.dumps(req))
```
