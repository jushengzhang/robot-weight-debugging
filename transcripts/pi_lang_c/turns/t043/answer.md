The broad category terms are indeed the risky part; the unambiguous aliases already have a causal dose-response. I will summarize paired outcomes without the large artifact paths and verify that the active embedding rows match recipe D exactly.
```python
torch.set_num_threads(16)
print("VARIANTS", type(variants).__name__, len(variants))
if isinstance(variants, dict):
    print("keys", list(variants)[:12])
    V = variants.get("variants", variants.get("practice_variants", []))
else:
    V = variants
print("SAMPLES", V[:2])

def pair_roll(k):
    ps = collections.defaultdict(dict)
    for r in read_roll(k):
        ps[(r["suite"], r["id"], r["seed"])][r["model"]] = r
    return ps

def summary(k):
    ps = pair_roll(k)
    valid = [p for p in ps.values() if all(m in p and isinstance(p[m].get("success"),bool) for m in ["original","edited"])]
    d = np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in valid])
    return {"n":len(valid), "listed":len(ps), "orig":sum(p["original"]["success"] for p in valid),
            "edit":sum(p["edited"]["success"] for p in valid), "gain_pp":round(100*d.mean(),2) if len(d) else None,
            "se_pp":round(100*d.std(ddof=1)/np.sqrt(len(d)),2) if len(d)>1 else None,
            "WR":int((d==1).sum()),"RW":int((d==-1).sum())}
print("PAIRED", [(k,summary(k)) for k in [14,15,16,17]])
print("\nD TARGETED")
for key,p in pair_roll(17).items():
    a,b = p["original"],p["edited"]
    print(key[:2], int(a["success"]), int(b["success"]), a["instruction"])
print("\nC CONFIRMATION DISCORDANCES")
for key,p in pair_roll(16).items():
    a,b = p["original"],p["edited"]
    if a["success"] != b["success"]:
        print(key[:2], int(a["success"]), int(b["success"]), a["instruction"])

E0 = load_original(E_NAME)
Ec = load(E_NAME)
changed = torch.nonzero(torch.any(Ec != E0, dim=1)).flatten().tolist()
print("\nCHANGED", len(changed), "recipe difference", sorted(set(changed)^set(D["rows"])))
for i in [65931,3425,31062,42856]:
    t = D["rows"][i]
    v = E0[t] if isinstance(t,int) else E0[t].mean(0)
    a = D["strengths"][i]
    linear = (1-a)*E0[i]+a*v
    u = v * (E0[i].norm()/v.norm())
    directional = (1-a)*E0[i]+a*u
    normed = directional*(E0[i].norm()/directional.norm())
    print(i, sp.id_to_piece(i), "norms", [round(float(x.norm()),3) for x in [E0[i],Ec[i],v]],
          "errors raw/directional/renorm", [round(float((Ec[i]-x).norm()),5) for x in [linear,directional,normed]])
del Ec
```
