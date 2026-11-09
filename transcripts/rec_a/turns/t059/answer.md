The earlier probe does not contain the attention activations needed for this decomposition. I will collect them on the same 16 inputs that already have residual and MLP recordings. This will let me separate the two residual branches using recorded values, without running any part of the network in the kernel.
```python
for k in (6,13):
    rec=[r for r in probe_records(k) if r["arm"]=="orig"]
    r=rec[0]
    print("Probe",k,"fields",list(r))
    if "activations" in r:
        a=np.load(local_path(r["activations"]))
        print("Call types",sorted(set(n.split("::")[1] for n in a.files if n.startswith("calls::"))))
    recmap={str(local_path(r["input"])):r for r in rec}
    print("P4 shared",sum(str(local_path(r["input"])) in recmap for r in p4))
print("P4 full input paths:")
print(json.dumps([str(local_path(r["input"])) for r in p4]))
with (W/"notes.md").open("a") as f:
    f.write("- E15H5 half-output probe: no sign changes; first-call downstream grip-axis RMS perturbation <0.75 vs natural signal std ~72. Reverted without rollout claim.\n"
            "- Next: separate adaptive attention and MLP residual branches using identical-input recorded activations.\n")
```
