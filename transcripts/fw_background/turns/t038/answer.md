I will compare each matched pair's attention-output change with its action-chunk change, and separate common texture-shift directions from task-specific variation. These are descriptive analyses only; an intervention is still needed to establish causality.

```python
probe3_acts={}
probe3_actions={}
def key_from_input_path(path):
    m=re.search(r"(libero_(?:10|goal|object|spatial))_v(\d+)_s(\d+)",str(path))
    return (m.group(1),int(m.group(2))) if m else None
for r in pr3:
    key=key_from_input_path(r["input"])
    with np.load(r["activations"],allow_pickle=False) as z:
        probe3_acts[key]=dict(z)
    probe3_actions[key]=np.load(r["action_norm"],allow_pickle=False)
first_probe_acts=next(iter(probe3_acts.values()))
print("Probe activation keys/shapes:",[(k,np.asarray(v).shape) for k,v in list(first_probe_acts.items())[:16]])
print("Number of activation keys:",len(first_probe_acts))
```
