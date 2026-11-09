The wrist-key edit’s gain is smaller than its standard error and includes a task regression, so I will remove it. I will return to the clearer environment diagnosis: matched scenes showed a dominant illumination shift. Instead of altering raw patch color responses, I will examine whether that nuisance remains identifiable at the frozen vision encoder’s output, where a more selective correction may be possible.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass20: G rejected\nPaired dev128:72->74 (+1.6pp SE2.2), env+1/object+1/position+1/task-1/semantic0; loops29->32. Wrist attention change only2-3pp in selected heads with opposite changes elsewhere. Removed. Note original dev baseline changed71->72 on this pass despite same sample/seed; use within-pass pairs, do not assume cross-pass exact reproducibility.\n")
print("roll1 activation path",r1[0].get("activations"))
p=r1[0].get("activations")
if isinstance(p,str):
    z=np.load(p)
    if hasattr(z,"files"):
        print("vision keys",[(k,z[k].shape) for k in z.files if "vision" in k][:45])
print("vision/projector tensors",[(n,SHAPES[n]) for n in NAMES if "post_layernorm" in n or "multi_modal_projector" in n])
env_pairs=[]
for i in range(0,len(ix14),2):
    pth=pathlib.Path(ix14[i]["input"])
    env=pathlib.Path(str(pth).replace("_semantic_","_environment_"))
    print(i//2,"environment exists",env.exists(),str(env).replace(str(W),"WORK"))
    if env.exists():env_pairs.extend([str(pth),str(env)])
(W/"environment_probe_inputs.json").write_text(json.dumps(env_pairs))
print("env pair inputs",len(env_pairs))
```
