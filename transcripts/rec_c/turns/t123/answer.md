The saved design separates the empty-closure signal from a reference signal used to center each late visual head. I will verify that the installed weights match that design and recover the original probe ordering so source states, reserved states, and open-gripper controls remain separate.
```python
for folder in ["probe_41", "probe_42"]:
    p = W/folder
    meta = [f for f in p.iterdir() if f.suffix == ".json"]
    print(folder, "metadata", [f.name for f in meta])
    for f in meta:
        obj = json.loads(f.read_text())
        print(f.name, json.dumps(obj, indent=1)[:3300])
m = json.loads((W/"value_contact_manifest.json").read_text())
print("manifest type", type(m).__name__, "len", len(m))
print("manifest first entries", json.dumps(m[:2] if isinstance(m,list) else m, indent=1)[:4000])
vname = "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.v_proj.weight"
cd = designs["centered_clearance_design.pt"]
actual_delta = load(vname) - load_original(vname)
print("delta relative to original", float(actual_delta.norm()/load_original(vname).norm()))
print("installed vs design error", float((actual_delta-cd["delta"]).norm()/cd["delta"].norm()))
print("candidate saved files", [p.name for p in (W/"candidate_centered_clearance_060").iterdir()])
print("fwd probe42 files", [p.name for p in (W/"fwd"/"probe_42").iterdir()] if (W/"fwd"/"probe_42").is_dir() else "not directory")
```
