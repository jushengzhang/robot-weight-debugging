Only the expected layer-17 attention-output tensor is edited. I will check its saved candidate metadata and the final panel against the supplied episode list, and inspect the rollout format so the final analysis can exclude simulator errors and account for both gripper starts sharing a scene.
```python
for filename in ("current_candidate.json", "final_candidate_protocol.json"):
    print("\n" + filename + ":")
    print((w / filename).read_text()[:2600])

panels = json.loads((w / "independent_panels.json").read_text())
def compact_schema(x, depth=0):
    if isinstance(x, dict):
        return {k: compact_schema(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        return {"length": len(x), "first": x[0] if x and depth < 3 else None,
                "last": x[-1] if x and depth < 3 else None}
    return x

print("\nPANEL STRUCTURE:")
print(json.dumps(compact_schema(panels), indent=2)[:3800])

for k in (48, 49):
    rows = [json.loads(line) for line in (w / f"roll_{k}.jsonl").read_text().splitlines() if line.strip()]
    print(f"\nROLL {k}: {len(rows)} records")
    print("First record:", json.dumps(rows[0])[:1900])
    error_rows = [r for r in rows if r.get("error") or r.get("success") is None]
    print("Potential error records:", len(error_rows))
    if error_rows:
        print(json.dumps(error_rows[:2])[:1700])

final_ids = [
    950, 935, 862, 787, 301, 160, 239, 427, 578, 496, 613, 567,
    820, 1191, 983, 1080, 1232, 1310, 1307, 1317, 197, 227,
    199, 464, 524, 603, 577, 688, 1007, 1066, 1285, 1332,
    1252, 1338, 9, 21, 13, 20, 44, 46, 38, 48, 124, 53, 84, 72,
]
final_episodes = [
    {"id": sid, "grip": grip, "seed": 29}
    for sid in final_ids for grip in ("open", "close")
]
print("\nFINAL PANEL:", len(final_episodes), "episodes,", len(set(final_ids)), "scenes")
```
