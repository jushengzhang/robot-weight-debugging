The controls succeed on all four tasks that failed under language perturbation, though their lighting differs. I will now build exact-observation instruction pairs and summarize the episode-level shifts only as a screening step; the paired probe will remove the visual and trajectory confounds.
```python
roll2=read_jsonl(W/"roll_2.jsonl")
control_by_task={(r["suite"],task_of(r)):r for r in roll2}
print("Control failure:",[(r["suite"],r["id"],r["instruction"]) for r in roll2 if not r["success"]])
print("Activation keys:",act0.files)
print("\nFailure action summaries (not semantic labels):")
for r in roll1:
    if r["success"]: continue
    a=np.load(r["actions_npy"])
    c=control_by_task[(r["suite"],task_of(r))]
    ac=np.load(c["actions_npy"])
    print(r["suite"],r["id"],"canon success",c["success"],
          "translation RMS %.3f/%.3f"%(np.sqrt(np.mean(a[:,:3]**2)),np.sqrt(np.mean(ac[:,:3]**2))),
          "gripper mean %.3f/%.3f"%(a[:,6].mean(),ac[:,6].mean()))
print("\nConfounded episode-mean residual shifts, fail vs success:")
for layer in (8,13,15,20,23,24,29):
    vals=[]
    for r in roll1:
        c=control_by_task[(r["suite"],task_of(r))]
        xa=np.load(r["activations"])
        xb=np.load(c["activations"])
        key=f"mean::video.{layer}.resid::agent"
        d=xa[key]-xb[key]
        vals.append((r["success"],float(d[1938]),float(np.linalg.norm(d)),
                     float(abs(d[1938])/max(np.linalg.norm(d),1e-9))))
    for success in (False,True):
        rows=[x for x in vals if x[0]==success]
        print("L%d %s delta1938=%.3f deltaL2=%.3f |1938|/L2=%.3f" %
              (layer,"success" if success else "failure",
               *np.mean([x[1:] for x in rows],axis=0)))
# Exactly identical saved images and proprio; only substitute an observed canonical prompt.
contrast_dir=W/"contrast_inputs"
contrast_dir.mkdir(exist_ok=True)
contrast_manifest=[]
for i,r in enumerate(roll1):
    src=Path(r["frames_dir"])/"t0000_inputs.pt"
    source=torch.load(src,map_location="cpu",weights_only=False)
    c=control_by_task[(r["suite"],task_of(r))]
    canonical=torch.load(Path(c["frames_dir"])/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    for condition in ("perturbed","canonical"):
        obj=dict(source)
        if condition=="canonical":
            obj["prompt"]=canonical["prompt"]
        dest=contrast_dir/f"{i:02d}_{condition[0]}.pt"
        torch.save(obj,dest)
        contrast_manifest.append({"pair":i,"condition":condition,"path":str(dest),
                                  "suite":r["suite"],"id":r["id"],"success":r["success"],
                                  "prompt":obj["prompt"]})
(W/"contrast_manifest.json").write_text(json.dumps(contrast_manifest,indent=2))
print("\nCreated",len(contrast_manifest),"paired replay inputs in",str(contrast_dir))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 2 controls\n"
            "13 same-task canonical-instruction lighting controls: 12/13 success. "
            "All four language-failure tasks succeed in controls. Lighting differs, "
            "so contrast_inputs contains 16 exact-observation pairs: original language "
            "input vs prompt copied from same-task practice control, identical pixels/proprio. "
            "Will probe with fixed diffusion noise. 2/40 passes used.\n")
```
