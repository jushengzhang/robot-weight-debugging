The simulator is not ready, so I will inspect the text projections and attention normalization weights while it starts. I am looking for localized outlier channels or heads—not a reason to weaken language conditioning globally—and will also identify comparable practice tasks.
```python
print("Non-block tensors:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, tuple(SHAPES[n]))
print("\nVideo block 0:")
for n in NAMES:
    if ".video.blocks.0." in n:
        print(n, tuple(SHAPES[n]))
print("\nCategory counts:", dict(collections.Counter(v["category"] for v in variants)))
print("Language examples:")
lang_variants = [v for v in variants if v["category"] == "Language Instructions"]
for s in sorted({v["suite"] for v in variants}):
    sv = [v for v in lang_variants if v["suite"] == s]
    print(s, len(sv), "difficulty", dict(collections.Counter(v["difficulty"] for v in sv)))
    for v in sv[:3]:
        print(v)
print("\nNormalization outlier summary (max absolute / median absolute):")
norm_stats = []
for n in NAMES:
    if (".cross_attn.norm_" in n or ".self_attn.norm_" in n) and n.endswith(".weight"):
        x = load_original(n)
        med = x.abs().median().item()
        norm_stats.append((x.abs().max().item()/max(med, 1e-8), n, med,
                           x.abs().max().item(), int(x.abs().argmax())))
for row in sorted(norm_stats, reverse=True)[:24]:
    print("%.3f %s median=%.4g max=%.4g index=%d" % row)
(W / "notes.md").open("a").write(
    "\n## Backend status\nInitial rollout refused while backend offline; no pass charged. "
    "Inspecting text-conditioning weights before retrying.\n")
```
