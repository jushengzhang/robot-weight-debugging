The five-row repair also failed to improve overall validation success, and its position losses are concerning. I will set it aside. A more precise language diagnosis is needed: earlier “word attention” included words such as “the” and “up,” so I will identify heads that actually read object names, destinations, and spatial qualifiers rather than generic prompt structure.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass19: five-row keep50% on validation128_a\n"
        "82->82/128, 0.0 +/-2.5pp; 5 wins/5 losses. Env13->16, object24->23, "
        "position11->8, semantic22->23, task12 unchanged. Loops27->20. "
        "Not a validated overall improvement; position damage. Set aside.\n"
    )
for n in list(edited()):
    drop_edit(n)

stop = {"", "<bos>", "<eos>", "<pad>", "\n", "the", "a", "an", "and", "it",
        "in", "on", "to", "of", "up", "pick", "place", "put", "into", "with",
        ".", ",", "please", "then", "grab", "take", "move"}
records_by_dir = {}
for rr in r1:
    records_by_dir[str(Path(rr["frames_dir"]).resolve())] = rr

attn_cases = defaultdict(list)
for pk in [2, 9, 13, 18]:
    js = json.loads((W/f"probe_{pk}"/"results.json").read_text())
    for rr in js:
        if rr["arm"] != "orig":
            continue
        rec = records_by_dir.get(str(Path(rr["input"]).parent.resolve()))
        if rec is None:
            continue
        text = rec["instruction"].lower().strip().replace("_", " ") + "\n"
        tids = sp.encode(text, add_bos=True)
        pieces = [sp.id_to_piece(t).replace(chr(9601), "").lower() for t in tids]
        nv = rr["prefix_layout"]["n_lang_valid"]
        if nv != len(tids):
            print("TOKEN LENGTH MISMATCH", rec["id"], nv, len(tids))
            continue
        content = np.array([p not in stop for p in pieces])
        for ls, path in rr["attention"].items():
            a = np.load(path).mean(1)
            attn_cases[int(ls)].append((rec["id"], rec["ptype"], pieces, content, a))

scores = []
for l, cases in sorted(attn_cases.items()):
    # Avoid duplicate observations across probe passes.
    unique = {}
    for c in cases:
        unique[(c[0], tuple(np.round(c[4][:,768], 6)))] = c
    cases = list(unique.values())
    for h in range(8):
        content_mass = [a[h,768:768+len(p)][mask].sum() for _,_,p,mask,a in cases]
        image_mass = [a[h,:512].sum() for *_,a in cases]
        scores.append((float(np.mean(content_mass)), l, h, len(cases), float(np.mean(image_mass))))
print("Top referent-content heads: (content_mean, layer, head, n, image_mean)")
for s in sorted(scores, reverse=True)[:18]:
    print(tuple(round(x,3) if isinstance(x,float) else x for x in s))
print("\nRequested content for wrong-task1946:")
for l, cases in sorted(attn_cases.items()):
    c = next((c for c in cases if c[0] == 1946), None)
    if not c:
        continue
    _,_,pieces,mask,a = c
    best_h = np.argsort([a[h,768:768+len(pieces)][mask].sum() for h in range(8)])[-2:][::-1]
    for h in best_h:
        t = a[h,768:768+len(pieces)]
        inds = np.argsort(t)[-4:][::-1]
        print("L%dH%d" % (l,h), "content", round(float(t[mask].sum()),3),
              [(pieces[i].encode("ascii","replace").decode(),round(float(t[i]),3)) for i in inds])
```
