The earlier head interventions changed attention but rarely changed gripper decisions, so they did not establish a repair mechanism. I will now examine where the late expert layers build the gripper signal and compare that pathway with the arm-motion signal.
```python
def local_path(s):
    s = str(s)
    if "/work/" in s:
        return W / s.split("/work/", 1)[1]
    return Path(s)

def probe_records(k):
    return json.loads((W / f"probe_{k}" / "results.json").read_text())

def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

p4 = probe_records(4)
r1 = read_roll(1)
print("Rollout fields:", list(r1[0]))
print("One episode:", {k:v for k,v in r1[0].items() if not isinstance(v, (list,dict))})
print("Selection:", (W / "gripper_probe_selection.json").read_text())
print("Diagnosis episodes format:", (W / "diagnosis_episodes.json").read_text()[:900])
print("Screen episodes:", (W / "screen_episodes.json").read_text()[:800])

for k in (4, 12, 13, 14):
    r = probe_records(k)[0]
    a = np.load(local_path(r["activations"]))
    print("Probe", k, "keys/shapes:", [(n, a[n].shape) for n in a.files][:45])
    print("Action:", np.load(local_path(r["action"])).shape)
    
inp = torch.load(local_path(p4[0]["input"]), map_location="cpu", weights_only=False)
print("Saved input fields:", {k: (tuple(v.shape), str(v.dtype)) if hasattr(v,"shape") else str(v)[:350] for k,v in inp.items()})
print("Expert tensor names around output:")
for n in NAMES:
    if ("gemma_expert.model.layers.17." in n or "gemma_expert.model.norm" in n
        or n.startswith("action_") or n.startswith("time_mlp")):
        print(n, SHAPES[n])
```
