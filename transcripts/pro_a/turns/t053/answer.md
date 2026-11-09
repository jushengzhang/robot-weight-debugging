Layer 12 head 2 is the strongest referent-reading head in the saved probes, but it reads the requested BBQ object much less than familiar objects. Rather than amplifying an entire head, I will estimate the instruction-contrast directions from matched task/paraphrase observations and separate them from common visual directions. This uses activation geometry only, not success labels or a fitted loss.
```python
for l in [4, 6, 12, 15]:
    print("Content diagnostics L", l)
    seen = set()
    for eid, pt, pieces, mask, a in attn_cases[l]:
        if eid in seen:
            continue
        seen.add(eid)
        hs = [2, 6] if l == 12 else [int(np.argmax([a[h,768:768+len(pieces)][mask].sum() for h in range(8)]))]
        for h in hs:
            t = a[h,768:768+len(pieces)]
            ids = np.argsort(t)[-4:][::-1]
            print(eid, pt, "H", h, "content", round(float(t[mask].sum()),3),
                  [(pieces[i].encode("ascii","replace").decode(),round(float(t[i]),3)) for i in ids])

mi = json.loads((W/"matched_initial.json").read_text())
semantic_inputs = [m for m in mi if m["ptype"] in ["semantic", "task"]]
assert len(semantic_inputs) == 32
assert all(Path(m["path"]).exists() for m in semantic_inputs)
(W/"semantic_contrast_inputs.json").write_text(json.dumps(semantic_inputs))
print("Contrast inputs:", json.dumps([m["path"] for m in semantic_inputs]))
print("Current edits:", edited())
with notes.open("a") as f:
    f.write(
        "\nRefined noun-content attention (excluding BOS/newline and generic syntax): "
        "L12H2 strongest mean30.2%, but task1946 only7.1%; L12H6 explicitly reads BBQ10.1% "
        "while BOS54.5%. Earlier L9H4 and L4 head boosts were not referent-specific. "
        "Plan an analytic contrast-subspace edit, using matched task/semantic initial observations "
        "(not identical images; visual contrasts will be measured and projected away). "
        "No success labels, regression, gradients, or loss fitting.\n"
    )
```
