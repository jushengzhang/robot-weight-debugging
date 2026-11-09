I’ll check whether the earlier action layers show both viewpoint-sensitive activations and unusually concentrated query/key coupling. Those two signals can identify a more specific upstream target than simply suppressing the final action output; neither signal alone is enough.
```python
# Inspect saved diagnostic formats and compact, weight/activation-only summaries.
print("Residual layer statistics:")
rz = np.load(root / "residual_diagnosis.npz")
print(np.round(rz["stats"], 4))
print("Attention-output layer statistics:")
az = np.load(root / "camera_firstcall_pca.npz")
print(np.round(az["stats"], 4))
for fn in ("all_head_qk_coupling.npz", "head_spectral_stats.npz", "weight_head_stats.npz"):
    z = np.load(root / fn)
    print(fn, {k: z[k].shape for k in z.files})
print("Weight coupling summary:", (root / "weight_qk_coupling.json").read_text()[:4700])

mz = np.load(root / "probe3_head_map_summary.npz")
for layer in (11, 16, 22, 29):
    mass = mz[f"L{layer}_video_mass"].mean(0)
    agent = mz[f"L{layer}_agent_fraction"].mean(0)
    ent = mz[f"L{layer}_entropy"].mean(0)
    tv = mz[f"L{layer}_camera_pair_tv"]
    inds = np.argsort(-mass*agent*(1-ent))[:5]
    print(f"L{layer} agent-concentrated heads (head, video_mass, agent_frac, entropy, pair_TV):",
          [(int(h), *np.round([mass[h], agent[h], ent[h], tv[h]], 3)) for h in inds])

for pname in ("probe_2", "probe_10", "probe_12"):
    idx = root / pname / "index.json"
    if idx.exists():
        obj = json.loads(idx.read_text())
        print(pname, "index:", str(obj)[:1000])
    p = root / pname / "probe_orig_0_acts.npz"
    if p.exists():
        z = np.load(p)
        print(pname, "activation shapes:", {k: z[k].shape for k in z.files[:5]})
print("Diagnostic group example:", str(json.loads((root / "diagnostic_groups.json").read_text()))[:1000])
```
