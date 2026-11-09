The easy goal instructions succeed in 17 of 18 episodes, compared with 31 of 64 difficult ones. I will now inspect exact tokenization and replay inputs, and compare task-feature similarity at the recorded layers before making any weight changes.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
r2 = read_roll(2)
def first_input(r):
    return str(pathlib.Path(r["frames_dir"])/"t0000_inputs.pt")
def structure(x):
    if isinstance(x, (torch.Tensor,np.ndarray)):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {k: structure(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [structure(v) for v in x[:3]]
    return str(x)[:120]
inp = torch.load(first_input(r1[0]), map_location="cpu", weights_only=False)
print("input structure", json.dumps(structure(inp), ensure_ascii=True)[:2300])
for r in r2:
    print(r["id"], int(r["success"]), r["instruction"])
for vid in [1415,1179,1339,1263,1331]:
    r=next(r for r in r1 if r["id"]==vid)
    ids=sp.encode(r["instruction"]+"\n",add_bos=True)
    print("tokens", vid, json.dumps(list(zip(ids,sp.id_to_piece(ids))),ensure_ascii=True))
def task_of(r):
    return r["name"].split("_language_")[0]
def cossim(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
a1={r["id"]:dict(np.load(r["activations"])) for r in r1}
a2={r["id"]:dict(np.load(r["activations"])) for r in r2}
for kind,dim in [("vlm","lang"),("expert","action")]:
    print("task nearest centroid",kind)
    for layer in [0,3,6,9,12,15,17]:
        key=f"mean::{kind}.{layer}.resid::{dim}"
        cent={t:np.mean([a2[r["id"]][key] for r in r2 if task_of(r)==t],axis=0) for t in tasks_goal}
        center=np.mean(list(cent.values()),axis=0)
        hit=collections.Counter()
        for r in r1:
            pred=max(cent,key=lambda t:cossim(a1[r["id"]][key]-center,cent[t]-center))
            hit[(r["success"],pred==task_of(r))]+=1
        print(layer,dict(hit))
print("probe candidates")
for vid in [1415,1390,1179,1169]:
    r=next(r for r in r1+r2 if r["id"]==vid)
    print(first_input(r).replace(str(W),"WORK"))
```
