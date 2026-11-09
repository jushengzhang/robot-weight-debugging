The full readout projection did not improve diagnostic success, so it is not accepted. The probe also showed that it reduced the large vertical background bias but increased the lateral discrepancy. I’ll test a narrower hypothesis: remove this texture feature only from the vertical-action readout, leaving lateral motion, rotation, and gripper outputs unchanged. I’ll test two doses and a norm-matched control before any independent validation.
```python
from collections import defaultdict

def paired_rows(pass_no):
    recs = [json.loads(s) for s in (W/f"roll_{pass_no}.jsonl").read_text().splitlines() if s.strip()]
    pairs = defaultdict(dict)
    for r in recs:
        pairs[(r["suite"],r["id"],r["seed"])][r["model"]] = r
    ans = []
    for key,p in pairs.items():
        if "original" in p and "edited" in p:
            o,e = p["original"],p["edited"]
            ans.append({"key":key,"category":o["category"],"difficulty":o.get("difficulty"),
                        "original":int(o["success"]),"edited":int(e["success"]),
                        "delta":int(e["success"])-int(o["success"]),
                        "original_steps":o["steps"],"edited_steps":e["steps"]})
    return ans

def paired_summary(rows):
    d = np.array([r["delta"] for r in rows],dtype=float)
    return {"n":len(rows),"original":sum(r["original"] for r in rows),
            "edited":sum(r["edited"] for r in rows),
            "gain_pp":float(100*d.mean()),
            "se_pp":float(100*d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None,
            "rescues":int((d>0).sum()),"regressions":int((d<0).sum())}

rows21 = paired_rows(21)
s21 = {c:paired_summary([r for r in rows21 if r["category"]==c])
       for c in sorted({r["category"] for r in rows21})}
print("Pass21:",s21)
print("Discordant episodes:",[(r["key"],r["category"],r["delta"]) for r in rows21 if r["delta"]])
(W/"roll21_summary.json").write_text(json.dumps(s21,indent=2))

# Same diagnosed feature, but only its existing vertical readout component.
u = torch.from_numpy(np.load(W/"readout_causal_directions.npz")["target"].copy())
H0 = load_original("mot.mixtures.action.head.weight")
zrow = H0[2]
zfree = zrow-P@(P.T@zrow)
e1 = zfree/zfree.norm()
g = torch.Generator().manual_seed(80513)
e0 = torch.randn(1024,generator=g)
S = torch.linalg.qr(torch.cat([P,e1[:,None]],1),mode="reduced").Q
e0 = e0-S@(S.T@e0)
e0 = e0/e0.norm()
mix = float(torch.abs(zrow@u)/zfree.norm())
assert 0 <= mix <= 1
control_z = mix*e1+math.sqrt(1-mix*mix)*e0
np.savez_compressed(W/"vertical_readout_directions.npz",
                    target=u.numpy(),control=control_z.numpy())

utility2 = '''
def set_vertical_readout(dose, kind="target"):
    W = Path(WORK)
    for name in list(edited()):
        drop_edit(name)
    direction = torch.from_numpy(np.load(W/"vertical_readout_directions.npz")[kind].copy())
    name = "mot.mixtures.action.head.weight"
    original = load_original(name)
    modified = original.clone()
    modified[2] -= dose*(original[2]@direction)*direction
    if dose:
        save_edit(name,modified)
    spec = {"family":"texture_feature_vertical_readout","dose":dose,"kind":kind,
            "tensor":name,"row":2,"rank":1,
            "relative_change":float((modified-original).norm()/original.norm())}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
'''
with (W/"experiment_utils.py").open("a") as f:
    f.write(utility2)
exec(utility2)
print("Applied:",set_vertical_readout(.5))
print("Matched control weight-norm ratio:",float(torch.abs(zrow@control_z)/torch.abs(zrow@u)))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass21 and narrowed causal hypothesis\n")
    f.write("- Full rank1 readout .5 paired diagnostic results: "+json.dumps(s21)+". No benefit, not accepted.\n")
    f.write("- Probe showed full projection lowers vertical BG-light discrepancy but increases lateral discrepancy. New exploratory candidate removes the same protected texture feature ONLY from vertical output row2; all other readout rows original.\n")
    f.write("- Preplanned tests: vertical doses .5 and1.0, then a norm-matched texture-orthogonal vertical control. Independent validation remains untouched.\n")
```
