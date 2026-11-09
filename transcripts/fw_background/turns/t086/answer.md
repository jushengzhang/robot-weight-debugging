The edit clearly softens attention, but its action changes are only 0.0007–0.0009 RMS and the texture-pair difference is essentially unchanged. That does not establish a useful control mechanism. I’ll remove it and inspect the saved feature-transfer analysis for components with stronger downstream influence.
```python
for n in list(edited()):
    drop_edit(n)
(W / "current_candidate.json").write_text(json.dumps({"family": "none", "reason": "query target effect near replay floor"}))

for fn in [
    "video_value_nuisance_info.json",
    "video_attention_contrast_info.json",
    "projection_diagnostic_info.json",
    "contrast_layer_statistics.json"
]:
    obj = json.loads((W / fn).read_text())
    print("\n", fn, type(obj).__name__)
    if isinstance(obj, dict):
        print("Keys:", list(obj)[:35])
        for k, v in list(obj.items())[:2]:
            print(k, str(v)[:1000])
    else:
        print(str(obj)[:1500])

for fn in [
    "video_nuisance_action_head_coupling.npz",
    "late_head_direct_readout_geometry.npz",
    "video_value_nuisance_data.npz",
    "video_attention_contrast_data.npz",
    "patch_weight_analysis.npz",
    "first_layer_spatial_mode_sensitivity.npz"
]:
    z = np.load(W / fn)
    print("\n", fn, [(k,z[k].shape) for k in z.files][:35])
    if fn == "video_nuisance_action_head_coupling.npz":
        for k in z.files:
            x = z[k]
            if x.ndim == 2:
                locs = np.dstack(np.unravel_index(np.argsort(x.ravel())[-12:][::-1],x.shape))[0]
                print(k, [(ij.tolist(), round(float(x[tuple(ij)]),5)) for ij in locs])
            else:
                print(k, x.tolist()[:20])
    if fn == "patch_weight_analysis.npz":
        for k in z.files:
            print(k, np.asarray(z[k]).reshape(-1)[:12].tolist())

print("\nExample saved activation keys:")
for k in [3,9,14]:
    fs = list((W / f"probe_{k}").glob("*acts.npz"))
    if fs:
        z = np.load(fs[0])
        print(k, [(q,z[q].shape) for q in z.files][:12])

with (W / "notes.md").open("a") as f:
    f.write("- Query-temperature dose .35 rejected as an actionable mechanism: attention entropy rises, but action shifts .0007-.0009 RMS are near replay noise; pair ratio .999816. Edits removed, no rollout spent.\n")
print("\nCurrent edits:", edited())
```
