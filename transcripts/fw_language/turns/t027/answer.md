The backend is now available. The small run has four failures, including two of three difficulty-5 episodes, but it is too small to estimate robustness. I will inspect the failure records and saved-input format, then obtain canonical-instruction controls for the same tasks.
```python
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
roll1=read_jsonl(W/"roll_1.jsonl")
print("Rollout row count:",len(roll1))
print("First row:",json.dumps(roll1[0],ensure_ascii=True)[:4500])
print("\nEpisode summaries:")
for r in roll1:
    print(json.dumps({k:v for k,v in r.items() if k not in
          ("activations","actions","trajectory","observations")},ensure_ascii=True)[:700])
idx1=json.loads((W/"roll_1_acts"/"index.json").read_text())
print("\nActivation index type:",type(idx1).__name__)
if isinstance(idx1,dict):
    print("Index keys:",list(idx1)[:10])
    print("Index preview:",json.dumps(idx1,ensure_ascii=True)[:2400])
else:
    print("Index preview:",json.dumps(idx1[:2],ensure_ascii=True)[:2400])
inputs1=sorted((W/"roll_1_frames").rglob("*_inputs.pt"))
print("\nSaved policy inputs:",len(inputs1))
inp1=torch.load(inputs1[0],map_location="cpu",weights_only=False)
def small_desc(x):
    if isinstance(x,torch.Tensor):
        return {"type":"tensor","shape":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,np.ndarray):
        return {"type":"array","shape":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,dict):
        return {str(k):small_desc(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return {"type":type(x).__name__,"len":len(x),"first":[small_desc(v) for v in x[:3]]}
    return str(x)[:500]
print("First input path:",str(inputs1[0]))
print("Input contents:",json.dumps(small_desc(inp1),ensure_ascii=True)[:7000])
with (W/"notes.md").open("a") as f:
    f.write("\n## Forward pass 1: initial diagnosis\n"
            "- Backend online. Original, 16 language episodes, seed 1701: "
            "12/16 successes (75%); difficulty 5: 1/3. Too small for conclusions.\n"
            "- Saved residuals, frames, and replayable inputs in roll_1_*.\n"
            "- 1/40 passes used; still no edits.\n")
```
