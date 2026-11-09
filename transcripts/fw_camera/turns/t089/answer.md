The saved residuals show viewpoint differences accumulating across many layers, not just the last few heads. I’ll inspect the activation resolution and the remaining unused practice variants before spending more passes. In particular, I want to distinguish a genuinely weak mechanism from an overly narrow two-direction edit.
```python
for pname in ("probe_2", "probe_3", "probe_10", "probe_12"):
    idx = root / pname / "index.json"
    if idx.exists():
        obj = json.loads(idx.read_text())
        print(pname, "index type:", type(obj).__name__,
              "sample:", str(obj if not isinstance(obj, list) else obj[:1])[:950])
    ps = sorted((root / pname).glob("*_acts.npz"))
    if ps:
        z = np.load(ps[0])
        print(pname, "activation fields:", [(k, z[k].shape) for k in z.files[:4]])
print("Stability:", (root / "camera_direction_stability.json").read_text())
print("Diagnostic groups:", str(json.loads((root / "diagnostic_groups.json").read_text()))[:1500])
print("Broad groups:", str(json.loads((root / "broad_diagnostic_groups.json").read_text()))[:1100])

practice_obj = json.loads((Path(BENCH) / "practice_variants.json").read_text())
print("Practice structure:", type(practice_obj).__name__)
if isinstance(practice_obj, dict):
    print("Practice keys:", list(practice_obj)[:10])
    practice = practice_obj.get("variants", practice_obj.get("practice_variants", []))
else:
    practice = practice_obj
seen_keys = set()
for p in root.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen_keys.add((r["suite"], r["id"]))
print("Practice total:", len(practice))
print("Camera unused:", sum(v["category"] == "Camera Viewpoints" and (v["suite"], v["id"]) not in seen_keys for v in practice))
print("Unused by category:", dict(__import__("collections").Counter(v["category"] for v in practice if (v["suite"], v["id"]) not in seen_keys)))
mz = np.load(root / "probe3_head_map_summary.npz")
qz = np.load(root / "all_head_qk_coupling.npz")
hs = np.load(root / "weight_head_stats.npz")
for layer, head in [(11,0),(11,2),(16,1),(22,5),(22,1),(29,4)]:
    print("Head", (layer,head),
          {k: round(float(qz[k][layer,head]),3) for k in qz.files},
          "o_norm", round(float(hs["action_o_head_norm"][layer,head]),3))
```
