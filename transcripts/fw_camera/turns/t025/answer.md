The backend is online. On the matched panel, the original model succeeded on 20/38 camera episodes versus 18/19 lighting controls. I will inspect the per-episode failures and activation summaries, then use first-observation probes to separate viewpoint effects from differences caused by later failed trajectories.```python
log_pass(1,"rollout","Original matched camera/light diagnosis",extra={"episodes":57})
roll1=[json.loads(line) for line in (W/"roll_1.jsonl").read_text().splitlines() if line.strip()]
print("Rollout record keys:",list(roll1[0]))
print("First two records:",json.dumps(roll1[:2],indent=2)[:6500])
act_index1=json.loads((W/"roll_1_acts"/"index.json").read_text())
print("Activation index type:",type(act_index1).__name__)
if isinstance(act_index1,dict):
    print("Index keys:",list(act_index1)[:12])
    first_key=next(iter(act_index1))
    print("First index value:",str(act_index1[first_key])[:2200])
else:
    print("First index entries:",str(act_index1[:2])[:2200])
frame_dirs=sorted(p for p in (W/"roll_1_frames").iterdir() if p.is_dir())
print("Frame directory count:",len(frame_dirs))
print("First directory:",frame_dirs[0].name)
print("First filenames:",[p.name for p in sorted(frame_dirs[0].iterdir())[:12]])
input0=sorted(frame_dirs[0].glob("*inputs.pt"))[0]
sample_input=torch.load(input0,map_location="cpu",weights_only=False)
def brief_structure(x,depth=0):
    if isinstance(x,torch.Tensor):
        return {"type":"tensor","shape":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,np.ndarray):
        return {"type":"array","shape":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,dict):
        return {str(k):brief_structure(v,depth+1) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return {"type":type(x).__name__,"len":len(x),"first":brief_structure(x[0],depth+1) if x else None}
    return str(x)[:180]
print("Saved input structure:",json.dumps(brief_structure(sample_input))[:2200])
del sample_input
with open(W/"notes.md","a") as f:
    f.write("\n## Pass 1: original matched diagnosis\n"
            "- Backend online. 57 episodes on 19 matched tasks/seeds.\n"
            "- Camera: 20/38 = 52.6%; light controls: 18/19 = 94.7%.\n"
            "- All-layer attention outputs and replayable observations saved in roll_1 artifacts.\n"
            "- Next: inspect layer shifts and first-observation attention; no edits yet.\n")
```
