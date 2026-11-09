Recipe D is active and changes exactly 66 rows. It fixes five of candidate C’s six known regressions, but its generic dairy representation loses several cheese-task recoveries. I will inspect the tokenization and trajectory summaries before deciding whether those ambiguous descriptions warrant any further edit.
```python
def ascii_piece(i):
    return sp.id_to_piece(int(i)).replace("\u2581","_")
def toks(s):
    return [(i,ascii_piece(i)) for i in sp.encode(s)]
for s in ["spreadable dairy product", "churned milk fat", "cultured milk", "red condiment container",
          "woven holder", "lift", "pick", "pick up", "spread", "cream cheese", "basin", "cooktop"]:
    print(s, toks(s))
print("\nTERMINATIONS")
for k in [7,9,10,12,14,15,16,17]:
    print(k, collections.Counter((r.get("termination"),r.get("success")) for r in read_roll(k)))
print("\nRESERVATIONS")
for p in sorted(W.glob("confirm_*episodes.json")):
    z=json.loads(p.read_text())
    print(p.name,len(z),z[:1])
print("Language count",collections.Counter(v["suite"] for v in V if v["category"]=="Language Instructions"))
print("\nFRAME CONTENTS")
fd = W/"roll_17_frames"/"libero_object_v1191_s7_original"
print([p.name for p in fd.iterdir()])
print("\nPROBE INDEX")
pi=json.loads((W/"probe_13"/"index.json").read_text())
print(str(pi)[:2300])
print("\nACT SCHEMA")
ap = W/"roll_17_acts"/"libero_object_v1191_s7_original.npz"
az=np.load(ap)
print([(k,az[k].shape) for k in az.files])
print("\nD CHANGED TOKENS IN REGRESSION")
reg=next(r for r in r17 if r["suite"]=="libero_object" and r["id"]==1191)
print([(ascii_piece(i),D["rows"][i],D["strengths"][i]) for i in sp.encode(reg["instruction"]) if i in D["rows"]])
print("\nDAIRY DESCRIPTIONS")
seen_ins=set()
for k in [1,5,16,17]:
    for r in read_roll(k):
        s=r["instruction"]
        if any(w in s for w in ["spread", "churned", "dairy", "cultured"]) and s not in seen_ins:
            seen_ins.add(s)
            task=r["name"].split("_language_")[0]
            print(r["suite"],r["id"],task,":",s)
```
