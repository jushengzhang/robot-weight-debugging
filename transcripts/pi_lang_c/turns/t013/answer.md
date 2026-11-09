I will compare centered task features across layers and examine the embedding geometry of the failure-associated synonyms. I will also build a protected-token set from all canonical task instructions, so a lexical edit can leave literal instructions—and therefore the other perturbation categories—unchanged.
```python
r5=read_roll(5)
print("Additional failures:")
for r in r5:
    if not r["success"]:
        print(r["suite"],r["id"],r["instruction"])
all_tasks=sorted(set(task_of(v) for v in lang))
def canonical_text(task):
    return re.sub(r"^[A-Z_]+SCENE\d+_", "", task).lower().replace("_"," ")
canonical_prompts=sorted(set(canonical_text(t) for t in all_tasks))
protected_ids=set([sp.bos_id(), sp.eos_id(), sp.pad_id()])
for t in canonical_prompts:
    protected_ids.update(sp.encode(t+"\n"))
print("\nCanonical task count",len(canonical_prompts),"protected token count",len(protected_ids))
print("Long-suite tasks:",[canonical_text(task_of(v)) for v in lang if v["suite"]=="libero_10"][:1],
      sorted(set(canonical_text(task_of(v)) for v in lang if v["suite"]=="libero_10")))
# Compare instruction-averaged residuals after centering on the ten task controls.
ctask={r["id"]:task_of(r) for r in r1}
for r in r2:
    ctask[r["id"]]=next((t for t in goal_tasks if r["name"].startswith(t+"_")),r["instruction"].replace(" ","_"))
short={1570:"bowl/stove",2473:"bowl/cabinet",2553:"wine/cabinet",2407:"plate/front",2521:"wine/rack",
       2490:"cheese/bowl",1785:"stove/on",74:"top/drawer+bowl",1828:"bowl/plate",2314:"middle/drawer"}
for group in ["vlm","expert"]:
    print("\nTask-centered nearest",group,"at layers 3/6/9/12/17:")
    for i in [1385,1420,1400,1262,1299,1179,1339,1127]:
        line=[]
        for l in [3,6,9,12,17]:
            key=f"mean::{group}.{l}.resid::"+("lang" if group=="vlm" else "action")
            X=np.stack([acts4[j][key] for j in control_ids])
            mu=X.mean(0); X=X-mu; x=acts4[i][key]-mu
            sim=(X@x)/(np.linalg.norm(X,axis=1)*np.linalg.norm(x)+1e-9)
            k=int(sim.argmax()); line.append(short[control_ids[k]]+":"+str(round(float(sim[k]),2)))
        print(i," | ".join(line))
EMB=next(n for n in NAMES if n.endswith("language_model.embed_tokens.weight"))
E0=load_original(EMB)
print("\nEmbedding norms / cosines:")
for a,b in [("cooker","stove"),("appliance","stove"),("basin","bowl"),("cupboard","cabinet"),
            ("compartment","drawer"),("grape","wine"),("vessel","bowl"),("dish","bowl")]:
    ia=sp.encode("the "+a)[-1]; ib=sp.encode("the "+b)[-1]
    va,vb=E0[ia],E0[ib]
    print(a,b,ia,ib,"norms",round(float(va.norm()),3),round(float(vb.norm()),3),
          "cos",round(float(torch.nn.functional.cosine_similarity(va,vb,dim=0)),3),
          "protected",ia in protected_ids)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 2-5\nLiteral goal controls 9/10. Pass 5 diagnostic hard-language sample 102/128; "
            "goal 27/44 vs object 22/23, spatial 16/18, long 37/43. "
            "Expert attention is not absent on failing instructions; BOS is a strong sink. "
            "Will test sparse synonym-embedding canonicalization, protecting all canonical task token IDs.\n")
```
