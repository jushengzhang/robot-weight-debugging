I will measure the replayed action changes separately for translation, rotation, and gripper commands. I will also check whether the downstream attention balance changed. This is a causal sanity check, not evidence of improved task success.
```python
r33 = json.loads((W / "probe_33/results.json").read_text())
print("Probe arms:", sorted({r["arm"] for r in r33}))
probe_pairs = {}
for r in r33:
    probe_pairs.setdefault(r["idx"], {})[r["arm"]] = r
diag33 = []
for idx, arms in sorted(probe_pairs.items()):
    ro = arms["orig"]
    re = next(v for k,v in arms.items() if k != "orig")
    a = np.load(local_path(ro["action"])).astype(float)
    b = np.load(local_path(re["action"])).astype(float)
    # Descriptive scalar projection only; not a weight fit or model change.
    scalar = float(np.sum(a[:,:6]*b[:,:6]) / max(np.sum(a[:,:6]**2),1e-12))
    nonglobal = float(np.linalg.norm(b[:,:6]-scalar*a[:,:6]) /
                      max(np.linalg.norm(b[:,:6]-a[:,:6]),1e-12))
    d = {"idx":idx, "best_descriptive_scale":round(scalar,4),
         "nonuniform_fraction_of_change":round(nonglobal,3),
         "xyz_rms_change":round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),4),
         "rot_rms_change":round(float(np.sqrt(np.mean((b[:,3:6]-a[:,3:6])**2))),4),
         "gripper_mean_orig":round(float(a[:,6].mean()),4),
         "gripper_mean_edit":round(float(b[:,6].mean()),4),
         "gripper_sign_changes":int(np.sum(np.sign(a[:,6])!=np.sign(b[:,6])))}
    diag33.append(d)
    print(d)
    for L in [16,17]:
        ao = np.load(local_path(ro["attention"][str(L)])).astype(float).mean(1)
        ae = np.load(local_path(re["attention"][str(L)])).astype(float).mean(1)
        delta = 100*(ae.sum(-1)-ao.sum(-1))
        print(" L", L, "prefix-mass change pp per head:", np.round(delta,2).tolist())
(W / "J_probe33_summary.json").write_text(json.dumps(diag33, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass33: J0.75 exact replay\n" +
            "Per-input descriptive action changes (no fitting used for edit):\n" +
            json.dumps(diag33) + "\n")
```
