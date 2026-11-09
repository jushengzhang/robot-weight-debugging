I will check whether the late-layer rank-one modes share an input direction or are unrelated features. A shared direction would motivate a narrow nuisance-subspace test; unrelated modes would argue against blanket spectral clipping. No weights will be changed during this screening.
```python
candidates = sorted(
    [r for r in all_spectral if r["expert"]=="video" and r["relative_s1"]>2.5],
    key=lambda r:r["relative_s1"], reverse=True)
modes = {}
for r in candidates:
    layer, head = r["layer"], r["head"]
    s,u,v = ov_spectrum("video",layer,head,vectors=True)
    modes[(layer,head)] = {"s":s, "u":u[:,0], "v":v[0]}
keys = list(modes)
U = torch.stack([modes[k]["u"] for k in keys])
V = torch.stack([modes[k]["v"] for k in keys])
cu,cv = (U@U.T).abs(),(V@V.T).abs()
pairs = [(float(cv[i,j]),float(cu[i,j]),keys[i],keys[j])
         for i in range(len(keys)) for j in range(i)]
print("Largest right-direction overlaps among outlier OV modes:")
for r in sorted(pairs,reverse=True)[:12]:
    print("input |cos|=%.3f output |cos|=%.3f heads=%s,%s" % r)
print("\nMatched-head singular gain in both experts:")
for r in candidates[:14]:
    a = next(x for x in all_spectral if x["expert"]=="action"
             and x["layer"]==r["layer"] and x["head"]==r["head"])
    print("L%02d H%02d video s1=%.2f frac=%.3f action s1=%.2f frac=%.3f" %
          (r["layer"],r["head"],r["s1"],r["frac"],a["s1"],a["frac"]))
print("\nPractice name examples for task matching:")
for cat in sorted({v["category"] for v in variants}):
    print(cat)
    for v in [x for x in variants if x["category"]==cat][:2]:
        print(" ",v["suite"],v["id"],v["name"])
torch.save({str(k):d for k,d in modes.items()},W/"candidate_video_OV_modes.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n- Exact head OV spectra reveal many video-specific late rank-one modes "
            "(e.g. L23 H3: s1=81.92, 97.7% energy; L24 H21: 64.77, 95.9%; "
            "L13 H7: 44.98, 76.5%). These are not evidence of a fault without activations.\n"
            "- No weights edited. Backend still offline after three free retries.\n")
```
