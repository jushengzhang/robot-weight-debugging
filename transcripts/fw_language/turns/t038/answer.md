The paired probe has completed. I will compare layerwise residual differences and action chunks, paying particular attention to whether the static channel-1938 hypothesis survives this controlled comparison. I will also inspect the artifact schema before requesting head-level attention maps.
```python
pr4=json.loads((W/"probe_4"/"results.json").read_text())
print("Probe result schema:",type(pr4).__name__)
print(json.dumps(pr4,ensure_ascii=True)[:3500])
# Normalize the result container without assuming a particular top-level key.
if isinstance(pr4,list):
    p4rows=pr4
else:
    p4rows=pr4.get("results",pr4.get("inputs",[]))
print("Rows:",len(p4rows) if isinstance(p4rows,list) else type(p4rows).__name__)
print("Probe files:",[p.name for p in (W/"probe_4").iterdir()][:12])
if isinstance(p4rows,list) and p4rows:
    row=p4rows[0]
    print("First row keys:",list(row))
    actpath=row.get("activations")
    if isinstance(actpath,str):
        p4a=np.load(actpath)
        print("Activation schema:",[(k,p4a[k].shape) for k in p4a.files[:12]],"total",len(p4a.files))
longpairs=sorted([p for p in natural_pairs if p["same_inputs"] and p["bad"]["suite"]=="libero_10"],
                 key=lambda p:p["action_rms"],reverse=True)[:2]
selected_pairs=selected_pairs+longpairs
(W/"natural_probe_pairs.json").write_text(json.dumps(selected_pairs,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 4 controlled natural-pair probe\n"
            "20 unmodified first-observation inputs: 10 same-task success/failure "
            "instruction pairs. All paired images and proprioception checked exactly "
            "equal; fixed diffusion noise seed7. Recorded all 30 residual layers. "
            "Includes 2 long-horizon task pairs. 4/40 used, no edits.\n")
```
