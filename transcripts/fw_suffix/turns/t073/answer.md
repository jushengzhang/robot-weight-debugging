I will compare this candidate’s action recovery and clean/paraphrase drift with the gate results, then move the more promising option toward rollout testing.
```python
m26=probe_metrics(26)
print(json.dumps(m26,indent=2))
(W/"candidate_oblique_text.json").write_text((W/"current_edit.json").read_text())
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 26: rank-24 oblique text removal\n"+json.dumps(m26)+"\n")
splits=json.loads((W/"splits.json").read_text())
print("Split keys and sizes:", {k:len(v) if hasattr(v,"__len__") else str(v) for k,v in splits.items()})
for k in ["development","validation_a","language_a"]:
    print(k,str(splits.get(k))[:900])
```
