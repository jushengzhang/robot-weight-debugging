The active tensor matches the frozen half-strength readout edit. I’ll load the next preplanned camera panel and check how the rollout records are organized, so I can combine the paired results correctly.
```python
panels = json.loads((root / "camera_validation_panels.json").read_text())
print("Panel structure:", type(panels).__name__)
if isinstance(panels, dict):
    print("Keys:", list(panels))
    print(str(panels)[:600])
else:
    print("Panel lengths:", [len(p) for p in panels])
    print("First item:", panels[0][0])
print("Analysis plan:", (root / "analysis_plan.json").read_text())
print("Pass ledger tail:", (root / "pass_ledger.json").read_text()[-2800:])
for k in (20, 21):
    path = root / f"roll_{k}.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    print(f"roll_{k}: {len(lines)} rows; first row:", json.dumps(lines[0])[:1800])
wname = "mot.mixtures.action.head.weight"
active_hash = hashlib.sha256(load(wname).contiguous().numpy().tobytes()).hexdigest()
print("Active tensor hash:", active_hash, "matches lock:", active_hash == json.loads(lock_path.read_text())["tensor_sha256"])
panel_c = panels[2] if isinstance(panels, list) else panels["panels"][2]
req22 = {"type": "rollout", "model": "both", "episodes": panel_c}
(root / "request_22.json").write_text(json.dumps(req22))
print("NEXT REQUEST:", json.dumps(req22, separators=(",", ":")))
```
