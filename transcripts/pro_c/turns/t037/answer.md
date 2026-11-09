Full illumination-component removal damages semantic controls, so I will revert it. The next diagnostic will isolate instruction content exactly: I will replay saved task observations with either their new instruction or the original scene instruction, keeping images and Gaussian noise identical. These are probes only, not additional simulator episodes.
```python
clear_edits()
contrast_dir=W/"instruction_contrasts"
contrast_dir.mkdir(exist_ok=True)
contrast_inputs=[]
for j,(suite,t,i) in enumerate(chosen_scenes):
    eid=by_scene[(suite,t)][i]["task"]["id"]
    src=pathlib.Path(rec1[eid]["frames_dir"])/"t0000_inputs.pt"
    dat=torch.load(src,map_location="cpu",weights_only=False)
    for label,prompt in [("old",rec1[eid]["instruction_orig"]),("new",rec1[eid]["instruction"])]:
        copied=dict(dat)
        copied["element"]=dict(dat["element"])
        copied["element"]["prompt"]=prompt
        out=contrast_dir/f"c{j:02d}_{label}.pt"
        torch.save(copied,out)
        contrast_inputs.append(str(out))
print("created",len(contrast_inputs),"matched probe inputs, prompt only changed")
print("relative paths:",[str(pathlib.Path(p).relative_to(W)) for p in contrast_inputs])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass13: reject full DC removal\nD1.0:71->69/128, -1.6pp SE2.9; semantic27->25. D0.5 remains only an unvalidated weak lead, not retained. All edits cleared.\n## Exact instruction counterfactuals\nCreated32 replay inputs from16 diagnostic task first observations: only prompt set to original scene instruction vs actual new instruction; identical pixels/noise. Probe-only, no extra simulator episodes. Goal: distinguish true instruction-sensitive subspace from BOS/no-op attention.\n")
```
