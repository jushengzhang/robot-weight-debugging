On the 48 tagged-category episodes, success rose from 26/48 to 36/48 when the suffix was removed—a 20.8-point gap. I will now verify the replay inputs and inspect the old gate’s construction, especially which token features it used and whether it actually edited both experts.
```python
def read_jsonl(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
roll1 = read_jsonl(W/"roll_1.jsonl")
roll2 = read_jsonl(W/"roll_2.jsonl")
print("ROLLOUT RECORD:", json.dumps(roll1[0])[:2500])
print("ACT INDEX:", (W/"roll_1_acts/index.json").read_text()[:1800])
input_paths = {}
for rid in (1,2):
    paths = sorted((W/f"roll_{rid}_frames").glob("*/t0000_inputs.pt"))
    input_paths[rid] = paths
    print("PASS",rid,"initial inputs",len(paths))
ip = torch.load(input_paths[1][0], map_location="cpu", weights_only=False)
print("INPUT TYPE:",type(ip).__name__)
if isinstance(ip,dict):
    for k,v in ip.items():
        print(" ",k,tuple(v.shape) if hasattr(v,"shape") else str(v)[:180],type(v).__name__)
prior_names = [p.stem for p in (W/"prior_candidate").glob("*.npy")]
print("PRIOR EDITS BY EXPERT:",dict(collections.Counter(n.split(".")[2] if n.startswith("mot.") else n for n in prior_names)))
print("PRIOR NON-BLOCK:", [n for n in prior_names if ".blocks." not in n])
for expert in ("action","video"):
    for part in ("text_embedding.0.weight","text_embedding.0.bias","text_embedding.2.weight","text_embedding.2.bias"):
        n=f"mot.mixtures.{expert}.{part}"
        pp=W/"prior_candidate"/f"{n}.npy"
        if pp.exists():
            delta=torch.from_numpy(np.load(pp))-load_original(n)
            if delta.ndim==2:
                rows=torch.where(delta.abs().amax(1)>1e-7)[0]
                cols=torch.where(delta.abs().amax(0)>1e-7)[0]
                print(expert,part,"changed rows",len(rows),rows[:12].tolist(),"cols",len(cols),cols[:12].tolist(),"norm",delta.norm().item())
            else:
                inds=torch.where(delta.abs()>1e-7)[0]
                print(expert,part,"changed",len(inds),inds[:12].tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n### Passes1-2: matched baseline\n48 tagged diagnostic: raw26/48=54.17%, strip36/48=75%; +20.83pp. Language16:10/16 both. Camera1->4/8; goal suite7->12/16 including language. Inputs and episode attn summaries saved.\n")
```
