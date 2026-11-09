Halving L15 H5 changed no gripper signs and produced only small action changes. Its effect is too small to explain the late gripper-signal amplification on its own. I will revert it and check whether the saved residual, attention, and MLP outputs can localize that amplification more precisely.
```python
for n in list(edited()):
    drop_edit(n)
for k in (4,6,12,13):
    rec=[r for r in probe_records(k) if r["arm"]=="orig"]
    r=rec[0]
    a=np.load(local_path(r["activations"]))
    types=sorted(set(n.split("::")[1] for n in a.files if n.startswith("calls::")))
    print("Probe",k,"inputs",len(rec),"call fields",types)
    print("first shape",a[[n for n in a.files if n.startswith("calls::")][0]].shape)
# Check exact shared-input correspondence; no fitting or model evaluation.
for k in (6,13):
    rec={str(local_path(r["input"])):r for r in probe_records(k) if r["arm"]=="orig"}
    print("P4 shared with",k,sum(str(local_path(r["input"])) in rec for r in p4))
print("P4 input paths:")
print(json.dumps([str(local_path(r["input"])) for r in p4]))
with (W/"notes.md").open("a") as f:
    f.write("- E15H5 half-output probe: no sign changes; first-call downstream grip-axis RMS perturbation <0.75 vs natural signal std ~72. Reverted without rollout claim.\n"
            "- Checking whether late adaptive MLP gating, rather than one self-attention head, builds the large gripper update.\n")
```
