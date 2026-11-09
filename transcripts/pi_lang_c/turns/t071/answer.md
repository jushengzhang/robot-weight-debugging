The fixed-input probe completed. I will check exact invariance for untouched prompts, verify that vision features are unchanged even on edited paraphrases, and quantify how VLM and action-expert representations move toward the literal-task controls.
```python
def probe_results(k):
    return json.loads((W/f"probe_{k}"/"results.json").read_text())
p30=probe_results(30)
print("RESULT SCHEMA",p30[0])
z=np.load(p30[0]["activations"])
print("ACT SHAPES",[(k,z[k].shape) for k in z.files])
P30={(r["idx"],r["arm"]):r for r in p30}
probe_cases[22]={"label":"layout_duplicate_object1997","path":probe_cases[21]["path"],"instruction":probe_cases[21]["instruction"]}
json.dump(probe_cases,(W/"F_controlled_probe_cases.json").open("w"),indent=2)
invariance=collections.Counter()
for i,c in enumerate(probe_cases):
    a,b=P30[(i,"orig")],P30[(i,"edit")]
    za,zb=np.load(a["activations"]),np.load(b["activations"])
    touched_i=bool(set(sp.encode(c["instruction"])) & set(F["rows"]))
    invariance[("action",touched_i,np.array_equal(np.load(a["action"]),np.load(b["action"])))]+=1
    for key in za.files:
        if "vision" in key:
            invariance[("vision",touched_i,np.array_equal(za[key],zb[key]))]+=1
        elif not touched_i:
            invariance[("untouched_activations",False,np.array_equal(za[key],zb[key]))]+=1
print("INVARIANCE",dict(invariance))
print("MATCHED CONTROL INPUT EQUALITY")
for i in range(0,12,2):
    a=torch.load(probe_cases[i]["path"],map_location="cpu",weights_only=False)
    b=torch.load(probe_cases[i+1]["path"],map_location="cpu",weights_only=False)
    print(probe_cases[i]["label"],"noise_equal",np.array_equal(a["noise"],b["noise"]),
          "base_equal",np.array_equal(a["element"]["observation/image"],b["element"]["observation/image"]),
          "wrist_equal",np.array_equal(a["element"]["observation/wrist_image"],b["element"]["observation/wrist_image"]))
```
