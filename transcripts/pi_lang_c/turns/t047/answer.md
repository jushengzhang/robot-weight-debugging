Removing “woven” did not change the success totals. I will check whether the same condiment regression remains; if so, “lift → pick” is its only remaining edited token. I will also inspect how the dairy descriptions distinguish cheese, butter, and milk rather than assigning “dairy” to cheese indiscriminately.
```python
p17,p18=pair_roll(17),pair_roll(18)
print("CHANGES D -> E")
for key in p17:
    if p17[key]["edited"]["success"] != p18[key]["edited"]["success"]:
        print(key,p17[key]["edited"]["success"],p18[key]["edited"]["success"])
print("REGRESSIONS E")
for key,p in p18.items():
    if p["original"]["success"] and not p["edited"]["success"]:
        s=p["original"]["instruction"]
        print(key,s,"edited tokens",[(ascii_piece(i),E["rows"][i]) for i in sp.encode(s) if i in E["rows"]])
print("\nDAIRY DESCRIPTIONS")
seen_ins=set()
for k in [1,5,16,17]:
    for r in read_roll(k):
        s=r["instruction"]
        if any(w in s for w in ["spread", "churned", "dairy", "cultured"]) and s not in seen_ins:
            seen_ins.add(s)
            print(r["suite"],r["id"],r["name"].split("_language_")[0],":",s)
print("\nTOKEN PIECES")
for word in ["spread","spreadable","cultured","churned","creamy","cream","cheese","butter","dairy","liquid"]:
    ids=[sp.piece_to_id(word),sp.piece_to_id("\u2581"+word)]
    print(word,[(i,ascii_piece(i),i in protected) for i in ids],toks("the "+word))
print("\nPROBE FILES", [p.name for p in (W/"probe_13").iterdir()][-8:])
print("ACT SCHEMA")
az=np.load(W/"roll_17_acts"/"libero_object_v1191_s7_original.npz")
print([(k,az[k].shape) for k in az.files])
print("INPUT SCHEMA")
inp = torch.load(W/"roll_17_frames"/"libero_object_v1191_s7_original"/"t0000_inputs.pt",map_location="cpu",weights_only=False)
print(type(inp).__name__, {k:(type(v).__name__,getattr(v,"shape",None)) for k,v in inp.items()})
```
