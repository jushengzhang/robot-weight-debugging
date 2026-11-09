I will measure expert attention to the individual words rather than assume that language is being ignored. I will also protect every token appearing in the canonical task names; leaving those embedding rows unchanged offers a concrete way to preserve the other perturbation categories.
```python
def piece_ascii(i):
    return sp.id_to_piece(int(i)).replace("\u2581","_")
for p in p3:
    saved=torch.load(p["input"],map_location="cpu",weights_only=False)
    text=saved["element"]["prompt"]
    ids=sp.encode(text+"\n",add_bos=True)
    print("\n",p["idx"],text,"action mean",np.round(np.load(p["action"]).mean(0),3).tolist())
    for layer,path in p["attention"].items():
        a=np.load(path).mean(1)
        lang=a[:,768:768+len(ids)]
        top=np.argsort(lang.mean(0))[-4:][::-1]
        masses=lang.sum(1)
        print(" L",layer,"language mass",np.round(masses,3).tolist(),
              "top",[(piece_ascii(ids[j]),round(float(lang[:,j].mean()),3)) for j in top])
import re
def canonical_from_name(name):
    s=name.split("_language_")[0]
    s=re.sub(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_","",s)
    return s.replace("_"," ").lower()
canon_texts=sorted({canonical_from_name(v["name"]) for v in langvars})
protected_ids=set()
for text in canon_texts:
    protected_ids.update(sp.encode(text+"\n",add_bos=True))
print("\ncanonical tasks",len(canon_texts),"protected tokens",len(protected_ids))
print("canonical tokens", [piece_ascii(i) for i in sorted(protected_ids)])
print("possible aliases")
aliases={"cooker":"stove","cook":"stove","cupboard":"cabinet","grape":"wine","grapes":"wine",
         "merlot":"wine","vessel":"bowl","rounded":"bowl","dairy":"cheese","holder":"rack",
         "storage":"cabinet","dish":"plate","dishware":"plate","citrus":"orange"}
for s,t in aliases.items():
    si=sp.encode("the "+s)[1:]; ti=sp.encode("the "+t)[1:]
    if len(si)==len(ti)==1:
        print(s,t,si[0],ti[0],"protected",si[0] in protected_ids,
              "cos",round(float(torch.nn.functional.cosine_similarity(E0[si[0]],E0[ti[0]],dim=0)),3))
```
