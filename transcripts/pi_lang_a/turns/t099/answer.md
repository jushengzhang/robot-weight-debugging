The probe output has matched original/edited records for all 40 inputs. I will check exact array equality, save the result, and load the first half of the precommitted third-seed hard-language check.
```python
probe_pairs = collections.defaultdict(dict)
for r in probe46:
    probe_pairs[int(r["idx"])][r["arm"]] = r
assert len(probe_pairs) == 40

probe_audit46 = {"inputs": 40, "actions_exact": 0, "all_recorded_activations_exact": 0,
                 "max_action_abs_difference": 0.0, "max_activation_abs_difference": 0.0,
                 "activation_arrays_compared": 0, "nonidentical_inputs": []}
for idx, pair in sorted(probe_pairs.items()):
    assert set(pair) == {"orig", "edit"}
    assert pair["orig"]["input"] == pair["edit"]["input"]
    ao = np.load(pair["orig"]["action"], allow_pickle=False)
    ae = np.load(pair["edit"]["action"], allow_pickle=False)
    assert np.isfinite(ao).all() and np.isfinite(ae).all()
    action_equal = np.array_equal(ao, ae)
    probe_audit46["actions_exact"] += int(action_equal)
    probe_audit46["max_action_abs_difference"] = max(
        probe_audit46["max_action_abs_difference"], float(np.max(np.abs(ao.astype(float)-ae))))
    acts_equal = True
    with np.load(pair["orig"]["activations"], allow_pickle=False) as zo, \
         np.load(pair["edit"]["activations"], allow_pickle=False) as ze:
        assert set(zo.files) == set(ze.files)
        for key in zo.files:
            x, y = zo[key], ze[key]
            assert np.isfinite(x).all() and np.isfinite(y).all()
            eq = np.array_equal(x, y)
            acts_equal &= eq
            probe_audit46["activation_arrays_compared"] += 1
            probe_audit46["max_activation_abs_difference"] = max(
                probe_audit46["max_activation_abs_difference"],
                float(np.max(np.abs(x.astype(float)-y))))
    probe_audit46["all_recorded_activations_exact"] += int(acts_equal)
    if not (action_equal and acts_equal):
        probe_audit46["nonidentical_inputs"].append(idx)
print("Canonical replay:", json.dumps(probe_audit46, separators=(",", ":")))
(W / "canonical_probe46_audit.json").write_text(json.dumps(probe_audit46, indent=2))

notes = (W / "notes.md").read_text()
notes = re.sub(r"Forward passes used: \d+/50\.", "Forward passes used: 46/50.", notes)
notes += "\nPass46 canonical fixed-input replay: " + json.dumps(probe_audit46) + ". Frozen18 unchanged.\n"
(W / "notes.md").write_text(notes)
print("REQUEST47")
print((W / "request47.json").read_text())
```
