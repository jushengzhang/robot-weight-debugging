The object-motion records show concrete instruction failures: for example, the model moves tomato sauce while the request names BBQ sauce, leaving the requested object untouched. I will compare attention in these failures with successful episodes from the same scenes, looking for heads that favor visual scene cues over instruction tokens.
```python
for task_fragment in ["tomato_sauce_and", "alphabet_soup_and", "salad_dressing_and", "put_the_bowl_on_the_stove"]:
    print("\nSCENE",task_fragment)
    for r in r1:
        if task_fragment in r["task"]:
            print(r["id"],r["ptype"],r["success"],r["orig_goal_reached"],r["grip_flips"],
                  r["instruction"][:90])
a0 = np.load(r1[0]["activations"])
print("\nActivation arrays:", [(k,a0[k].shape) for k in list(a0)[:8]])
print("Norms by stream/layer (baseline episode means):")
for typ in ["object","position","semantic","task","environment"]:
    rr=[r for r in r1 if r["ptype"]==typ]
    out={}
    for l in [0,6,12,17]:
        for stream in ["vlm","expert"]:
            keys = [k for k in a0.files if f"{stream}.{l}." in k]
            out[f"{stream}{l}"] = {
                k.split("::")[-1]: round(float(np.mean([np.linalg.norm(np.load(r["activations"])[k]) for r in rr])),1)
                for k in keys}
    print(typ,out)
# Save robust helpers for exact diagnostic replay.
def first_input(r):
    return str(sorted(pathlib.Path(r["frames_dir"]).glob("*inputs.pt"))[0])
target = [next(r for r in r1 if r["id"]==1946),
          next(r for r in r1 if r["task"]==byid[1946]["task"] and r["success"] and r["ptype"] in ["object","semantic"]),
          next(r for r in r1 if r["id"]==1327),
          next(r for r in r1 if r["task"]==byid[1327]["task"] and r["success"] and r["ptype"] in ["object","semantic"])]
diag_inputs = [first_input(r) for r in target]
print("\nProbe cases:", [(r["id"],r["ptype"],r["success"]) for r in target])
print("Probe paths:", json.dumps(diag_inputs))
(W/"diag_inputs.json").write_text(json.dumps(diag_inputs))
```
