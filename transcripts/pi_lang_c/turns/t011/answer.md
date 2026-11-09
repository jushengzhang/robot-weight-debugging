I will quantify which heads read language, which tokens receive their attention, and how close each failed action chunk is to the literal task references. I will also compare task-centered residual features rather than raw cosine similarity, which can be dominated by features shared across all tasks.
```python
def input_id(e):
    return int(re.search(r"_v(\d+)_", e["input"]).group(1))
def token_pieces(prompt):
    return ["<bos>"] + sp.id_to_piece(sp.encode(prompt.strip().lower().replace("_"," ")+"\n"))
for e in e3:
    data=torch.load(e["input"],map_location="cpu",weights_only=False)
    pcs=token_pieces(data["element"]["prompt"])
    print("\nID",input_id(e),data["element"]["prompt"],"tokens:",list(enumerate(pcs)))
    for l,path in e["attention"].items():
        at=np.load(path).mean(axis=1)
        lm=at[:,768:968].sum(axis=1)
        print("L",l,"language mass/head",np.round(lm,3).tolist())
        topheads=np.argsort(lm)[-2:][::-1]
        for h in topheads:
            t=np.argsort(at[h,768:768+len(pcs)])[-3:][::-1]
            print(" H",int(h),"top",[(pcs[j],round(float(at[h,768+j]),3)) for j in t])
z=np.load(e4[0]["activations"])
print("\nProbe activation keys/shapes:",[(k,z[k].shape) for k in z.files])
acts4={input_id(e):np.load(e["activations"]) for e in e4}
actions4={input_id(e):np.load(e["action"]) for e in e4}
control_ids=[r["id"] for r in r2]
task_labels={r["id"]:r["instruction"] for r in r2}
for e in e4[:11]:
    i=input_id(e)
    dist=[np.sqrt(np.mean((actions4[i][:5,:3]-actions4[j][:5,:3])**2)) for j in control_ids]
    order=np.argsort(dist)[:3]
    print("Action nearest",i,[(j:=control_ids[k],task_labels[j],round(float(dist[k]),4)) for k in order])
```
