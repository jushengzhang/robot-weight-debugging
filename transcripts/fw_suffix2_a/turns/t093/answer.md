The standard gate failed this development safety check: clean instructions lost 3 of 80 successes, and paraphrases lost 3 of 48. I won’t treat it as a validated repair. I’ll inspect the exact weight changes to distinguish detector false positives from collateral changes caused by wiring the gate into normalized attention.
```python
print("Pass27:", paired_rollout(27))
names = list(edited())
block0 = [n for n in names if ".blocks.0." in n]
print("Block0 edited names:", block0)
for n in block0 + [n for n in names if "text_embedding" in n]:
    a, b = load_original(n), load(n)
    dif = b-a
    nz = dif != 0
    print(n, "shape", list(a.shape), "changed", int(nz.sum()),
          "relative_norm", round(float(dif.norm()/a.norm()),5))
    if a.ndim == 1:
        inds = torch.where(nz)[0]
        print("  first positions/old/new:", [(int(j),round(float(a[j]),4),round(float(b[j]),4)) for j in inds[:8]])
    elif "cross_attn" in n:
        rows = torch.where(nz.any(dim=1))[0]
        print("  changed rows", len(rows), "first row original/edit norms",
              [(int(j),round(float(a[j].norm()),4),round(float(b[j].norm()),4)) for j in rows[:4]])
print("\nSCANNER DEFINITION (first 3500 chars):")
print((W / "dense_readout.py").read_text()[:3500])
with (W / "notes.md").open("a") as f:
    f.write("\n### Pass27 standard video-gate development safety FAILED\n"+
            json.dumps(paired_rollout(27))+
            "\nClean60->57/80 (-3.75pp); paraphrase30->27/48(-6.25pp). "
            "Do not retain this gate without correcting collateral damage and independent validation.\n")
```
