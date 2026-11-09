The late self-attention heads are active in successful episodes too, so their presence alone does not identify a fault. Layer 16 head 3 is more distinctive: it reads instruction content while most neighboring heads read action tokens. I will check which words it reads before deciding whether to test its contribution.
```python
for pn in [2, 31]:
    rr = json.loads((W / f"probe_{pn}/results.json").read_text())
    print("\nPROBE", pn)
    for r in rr:
        inp = torch.load(local_path(r["input"]), map_location="cpu", weights_only=False)
        prompt = inp["element"]["prompt"].strip().replace("_", " ").replace("\n", " ")
        ids = sp.encode(prompt + "\n", add_bos=True)
        nv = r["prefix_layout"]["n_lang_valid"]
        tokens = [sp.id_to_piece(i).replace("\u2581", "_") for i in ids]
        print(r["idx"], inp["ptype"], "step", inp["step"], repr(prompt), "tokens", len(ids), nv)
        for L, h in [(16,3), (16,6), (17,1)]:
            if str(L) not in r["attention"]:
                continue
            aa = np.load(local_path(r["attention"][str(L)])).astype(float)[h].mean(0)
            inds = np.argsort(aa[768:768+nv])[::-1][:5]
            top = [(tokens[i] if i < len(tokens) else f"token{i}", round(100*aa[768+i],1)) for i in inds]
            print(" L", L, "h", h, "prefix%", round(100*aa.sum(),1), "top words", top)

print("\nRelevant weight shapes:")
for name in NAMES:
    if ("gemma_expert.model.layers.16.self_attn." in name or
        "gemma_expert.model.layers.17.self_attn.o_proj" in name or
        "action_out_proj" in name):
        print(name, SHAPES[name])
Oname = "paligemma_with_expert.gemma_expert.model.layers.16.self_attn.o_proj.weight"
O16 = load_original(Oname)
Aout = load_original("action_out_proj.weight")
print("Head O norms and normalized direct output alignment (not a causal attribution):")
for h in range(8):
    b = O16[:, h*256:(h+1)*256]
    align = (Aout @ b).norm(dim=1) / (Aout.norm(dim=1) * b.norm()).clamp_min(1e-10)
    print(h, "O norm", round(float(b.norm()),2), "alignment first7",
          [round(float(x),4) for x in align[:7]])

dev = json.loads((W / "dev_episodes.json").read_text())
print("\nDev manifest structure:", type(dev).__name__, "preview", str(dev)[:700])
pr = json.loads((Path(BENCH) / "practice_episodes.json").read_text())
print("Practice list structure", type(pr).__name__, "length", len(pr))
if isinstance(pr, dict):
    print("Practice keys", list(pr))
else:
    from collections import Counter
    print("Practice perturbation counts", dict(Counter(x["ptype"] for x in pr)))
```
