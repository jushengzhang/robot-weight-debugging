The broader sample again points to the goal suite as the main weakness. I will quantify which language tokens the expert attends to and compare the first predicted actions, then inspect token embeddings for the failed substitutions.
```python
p3=json.loads((W/"probe_3"/"results.json").read_text())
print("probe structure", str(type(p3)))
print(json.dumps(p3,ensure_ascii=True)[:4400])
r4=read_roll(4)
print("broad failures")
for r in r4:
    if not r["success"]:
        print(r["suite"],r["id"],r["difficulty"],r["instruction"])
print("embedding token examples")
Ename=next(n for n in NAMES if n.endswith("language_model.embed_tokens.weight"))
E0=load_original(Ename)
for text in ["stove","cooktop","cooker","appliance","cabinet","cupboard","storage","unit","bowl","container","dish","plate","wine","grape","fermented","dairy","cream","cheese","rack","holder"]:
    ids=sp.encode("the "+text)[1:]
    print(text,[(i,round(E0[i].norm().item(),3)) for i in ids])
```
