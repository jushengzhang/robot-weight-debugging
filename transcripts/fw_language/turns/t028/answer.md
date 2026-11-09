The failed instructions include noun substitutions and an implicit prerequisite (“store the bowl” instead of explicitly opening the drawer), so this is not simply a prompt-length problem. I will inspect only the compact replay and activation schemas next, then select same-task controls without changing observations or noise in the probes.
```python
print("Failures:")
for r in roll1:
    if not r["success"]:
        print(r["suite"],r["id"],r["seed"],r["instruction"])
print("\nActivation index preview:",json.dumps(idx1,ensure_ascii=True)[:1100])
print("\nFirst saved input:",str(inputs1[0].relative_to(W)))
print("Top-level type:",type(inp1).__name__)
if isinstance(inp1,dict):
    for k,v in inp1.items():
        d=small_desc(v)
        print(k, json.dumps(d,ensure_ascii=True)[:850])
print("\nActivation array schema:")
act0=np.load(roll1[0]["activations"])
print([(k,act0[k].shape) for k in act0.files[:8]])
print("Number of arrays:",len(act0.files))
print("\nActions schema:",np.load(roll1[0]["actions_npy"]).shape)
# Select one low-difficulty canonical-instruction lighting control per diagnostic task.
control_eps=[]
control_map={}
for r in roll1:
    task=r["name"].split("_language_")[0]
    choices=by_task_cat[(r["suite"],task,"Light Conditions")]
    c=min(choices,key=lambda v:(v["difficulty"],v["id"]))
    key=(c["suite"],c["id"],r["seed"])
    if key not in control_map:
        control_map[key]=c
        control_eps.append({"suite":c["suite"],"id":c["id"],"seed":r["seed"]})
print("\nMatched low-light controls:",len(control_eps))
print(json.dumps(control_eps,separators=(",",":")))
(W/"diagnostic_control_episodes.json").write_text(json.dumps(control_eps))
```
