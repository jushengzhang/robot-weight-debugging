I will summarize H15's attention and action changes, then restore the byte-verified I065 checkpoint for the same four-input audit. Neither result will change the decision to reject candidates with no pooled validation gain.
```python
audit_labels = ["late_loop_65", "recovery_success_65", "prefail_success_735", "prefail_success_744"]
audit_heads = [(2, 0), (7, 2), (15, 1), (16, 6)]

def summarize_probe(k):
    root = W / f"probe_{k}"
    rows = []
    for i, label in enumerate(audit_labels):
        ao = np.load(root / f"probe_orig_{i}_action.npy")
        ae = np.load(root / f"probe_edit_{i}_action.npy")
        row = {
            "input": label,
            "translation_rms_change": float(np.sqrt(np.mean((ae[:, :3] - ao[:, :3])**2))),
            "rotation_rms_change": float(np.sqrt(np.mean((ae[:, 3:6] - ao[:, 3:6])**2))),
            "gripper_rms_change": float(np.sqrt(np.mean((ae[:, 6] - ao[:, 6])**2))),
            "original_first5_gripper": float(ao[:5, 6].mean()),
            "heads": {}
        }
        print(label, {k: round(v, 5) for k, v in row.items() if isinstance(v, float)})
        for layer, head in audit_heads:
            masses = {}
            for arm in ("orig", "edit"):
                a = np.load(root / f"probe_{arm}_{i}_attn_L{layer}.npy")
                assert a.shape == (8, 10, 968), a.shape
                p = a[head].mean(axis=0)
                masses[arm] = {
                    "base": float(p[:256].sum()),
                    "wrist": float(p[256:512].sum()),
                    "prompt": float(p[768:].sum()),
                    "self": float(1.0 - p.sum())
                }
            row["heads"][f"L{layer}h{head}"] = masses
            print(f"  L{layer}h{head}", {
                arm: {src: round(v, 3) for src, v in m.items()}
                for arm, m in masses.items()
            })
        rows.append(row)
    return rows

audit49 = summarize_probe(49)
(W / "final_H15_routing_audit.json").write_text(json.dumps(audit49, indent=2))
print("Probe metadata preview:", (W / "probe_49" / "results.json").read_text()[:1400])

for name in list(edited()):
    drop_edit(name)
for p in sorted((W / "candidate_I065").glob("*.npy")):
    name = p.name[:-4]
    assert name in NAMES
    save_edit(name, torch.from_numpy(np.load(p)))
assert len(edited()) == 3
print("I065 restored:", edited())
for p in sorted((W / "candidate_I065").glob("*.npy")):
    q = W / "edited" / p.name
    print(p.name, "exact_saved_match", np.array_equal(np.load(p), np.load(q)))

with (W / "notes.md").open("a") as f:
    f.write(
        "\n## Final routing audit, pass49\n"
        "Probed frozenH15 on4 same-instruction/open-command inputs:65open t300 loop,"
        "65close t50 successful recovery,735open/744open seed7 t0 successful pre-fail."
        "Layer2/7/15/16 head-source masses and action changes saved in "
        "final_H15_routing_audit.json. This small audit is not a new efficacy test. "
        "FrozenI065 restored exactly for the same-input final pass.\n"
    )
```