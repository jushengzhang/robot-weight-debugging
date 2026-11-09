Strengthening the two instruction-reading heads does not help and slightly damages controls, so I will revert it. I will next test a different hypothesis: later MLP updates may overwrite instruction-sensitive features with a scene-conditioned action pattern. First I will verify how closely the matched initial observations and noise agree.
```python
clear_edits()
print("matched initial inputs: image MAE, wrist MAE, noise RMS difference")
for suite,t,i in chosen_scenes:
    g=by_scene[(suite,t)][i]
    oo=torch.load(pathlib.Path(rec1[g["semantic"]["id"]]["frames_dir"])/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    tt=torch.load(pathlib.Path(rec1[g["task"]["id"]]["frames_dir"])/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    vals=[np.abs(oo["element"][k].astype(float)-tt["element"][k].astype(float)).mean()
          for k in ["observation/image","observation/wrist_image"]]
    vals.append(np.sqrt(np.mean((oo["noise"]-tt["noise"])**2)))
    print(g["task"]["id"],[round(float(x),4) for x in vals])
print("initial expert residual task/semantic norm-relative delta for four probed scenes:")
for l in [0,4,8,12,16,17]:
    vals=[]
    for i in [0,4,8,12]:
        a=np.load(p3[i]["activations"])[f"calls::expert.{l}.resid::action"]
        b=np.load(p3[i+1]["activations"])[f"calls::expert.{l}.resid::action"]
        vals.append(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-9))
    print(l,[round(float(x),4) for x in vals])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass11: reject candidate C1\nL4 expert instruction-head O blocks3/7 x1.5:71->70/128, -0.8pp SE2.6. Task8->6, object18->17, semantic27->26; insufficient/negative. All edits cleared. Next localize whether MLPs overwrite instruction-sensitive residual components.\n")
```
