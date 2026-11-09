I’ll summarize the attention maps and confirm that the restored checkpoint reproduces its earlier probe outputs exactly. I’ll also fix the remaining test plan now: six difficult non-language audits, followed by 384 previously reserved language variants at a third explicit rollout seed. No further weight changes are planned.
```python
r41 = json.loads((W/"probe_41"/"results.json").read_text())
a41 = {(r["idx"], r["arm"]): r for r in r41}
row_set = set(row_ids)
attn_summary = []

for i, old_i in enumerate([0, 3, 4, 5]):
    rec = a41[i, "orig"]
    data = torch.load(rec["input"], map_location="cpu", weights_only=False)
    prompt = data["element"]["prompt"]
    toks = sp.encode(prompt.strip().replace("_", " ").replace("\n", " ") + "\n", add_bos=True)
    layout = rec["prefix_layout"]
    n_valid = layout["n_lang_valid"]
    assert len(toks) == n_valid, (len(toks), n_valid)
    start = layout["lang"][0]
    word_pos = list(range(start+1, start+n_valid-1))
    changed_pos = [start+j for j, t in enumerate(toks) if t in row_set]
    print("\nInput", i, "tokens", [ascii_piece(t) for t in toks],
          "changed positions", [p-start for p in changed_pos])
    print("Restored action matches previous full checkpoint:",
          np.array_equal(np.load(a41[i, "edit"]["action"]), action(23, old_i, "edit")))
    heads = []
    for layer in [0, 3, 9, 15]:
        maps = {arm: np.load(a41[i, arm]["attention"][str(layer)]).astype(np.float64)
                for arm in ["orig", "edit"]}
        stats = {}
        for arm, m in maps.items():
            stats[arm] = {
                "base": float(m[:, :, :256].sum(-1).mean()),
                "wrist": float(m[:, :, 256:512].sum(-1).mean()),
                "valid_language": float(m[:, :, start:start+n_valid].sum(-1).mean()),
                "content_words": float(m[:, :, word_pos].sum(-1).mean()),
                "edited_words": float(m[:, :, changed_pos].sum(-1).mean()) if changed_pos else 0.0,
            }
        attn_summary.append({"input": i, "layer": layer, **stats})
        if i in [0, 2]:
            print("L", layer, {arm: {k: round(v, 4) for k, v in st.items()} for arm, st in stats.items()})
            for h in range(8):
                old_mass = float(maps["orig"][h, :, changed_pos].sum()/10) if changed_pos else 0
                new_mass = float(maps["edit"][h, :, changed_pos].sum()/10) if changed_pos else 0
                heads.append((old_mass, layer, h, new_mass))
        else:
            assert np.array_equal(maps["orig"], maps["edit"]), "Untouched control attention changed"
    if heads:
        print("Highest original edited-token attention heads (old mass, layer, head, new mass):",
              [(round(a, 4), b, c, round(d, 4)) for a,b,c,d in sorted(heads, reverse=True)[:5]])

(W/"causal_controls"/"attention_summary.json").write_text(json.dumps(attn_summary, indent=2))
manifest = json.loads((W/"validation_manifest.json").read_text())
requests73 = [
    {"type": "rollout", "model": "both",
     "episodes": [{"suite": r["suite"], "id": r["id"], "seed": 73} for r in batch]}
    for batch in manifest[:3]
]
dev_set = {(r["suite"], r["id"]) for rr in records.values() for r in rr}
assert all((e["suite"], e["id"]) not in dev_set for req in requests73 for e in req["episodes"])
(W/"validation_seed73_requests.json").write_text(json.dumps(requests73, indent=2))
print("\nFrozen third-seed validation:", [len(r["episodes"]) for r in requests73], "seed73",
      "development overlap: 0")
with (W/"notes.md").open("a") as f:
    f.write("\nPass41 final-checkpoint attention replay: all4 edited action chunks exactly match prior probe23; "
            "both literal controls have bit-identical original/edited attention maps at all4 layers. "
            "Detailed masses saved causal_controls/attention_summary.json. "
            "Third-seed requests frozen in validation_seed73_requests.json (first384 reserved variants, "
            "explicit seed73, zero overlap with development).\n")
```
