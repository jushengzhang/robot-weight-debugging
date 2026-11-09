The probe reports action-to-video maps, not text-token maps, so it cannot directly identify heads attending to suffix tokens. I will use those maps to track the indirect visual pathway, inspect the returned activations, and prepare exact matched-input probes for causal edits to each expert’s text cross-attention.
```python
probe3 = json.loads((work / "probe_3/results.json").read_text())
print("Probe result structure:", str(probe3)[:5500])
p3files = list((work/"probe_3").rglob("*"))
print("Probe file examples:", [str(p.relative_to(work/"probe_3")) for p in p3files[:15]])
npzs = [p for p in p3files if p.suffix == ".npz"]
if npzs:
    with np.load(npzs[0]) as a:
        print("Probe activation keys/shapes:", [(k,a[k].shape) for k in a.files[:15]])

diag_inputs_dir = work / "diag_inputs"
diag_inputs_dir.mkdir(exist_ok=True)
input_manifest = []
initial_diff = []
for i,k in enumerate(common):
    rp = Path(rmap[k]["frames_dir"])/"t0000_inputs.pt"
    cp = Path(cmap[k]["frames_dir"])/"t0000_inputs.pt"
    r = torch.load(rp, map_location="cpu", weights_only=False)
    c = torch.load(cp, map_location="cpu", weights_only=False)
    paths = {}
    for kind, source in [("r",rp),("c",cp)]:
        target = diag_inputs_dir / f"{kind}{i:02d}.pt"
        if not target.exists():
            target.symlink_to(source)
        paths[kind] = "WORK/diag_inputs/" + target.name
    input_manifest.append({"pair":i, "key":k, "category":rmap[k]["category"],
                           "raw":paths["r"], "clean":paths["c"],
                           "raw_prompt":r["prompt"],"clean_prompt":c["prompt"]})
    initial_diff.append({
        "key":k, "category":rmap[k]["category"],
        "image_equal":torch.equal(r["input_image"],c["input_image"]),
        "proprio_equal":torch.equal(r["proprio"],c["proprio"]),
        "action_rms":float(np.sqrt(np.mean((np.asarray(r["action_chunk"])-np.asarray(c["action_chunk"]))**2))),
        "raw_success":rmap[k]["success"],"clean_success":cmap[k]["success"]})
print("Initial obs equality:", sum(x["image_equal"] for x in initial_diff), "/24 images;",
      sum(x["proprio_equal"] for x in initial_diff), "/24 proprio")
print("Initial action RMS differences by category:")
for cat in tag_cats:
    vals = [x["action_rms"] for x in initial_diff if x["category"]==cat]
    print(cat, round(float(np.mean(vals)),4))
print("Probe inputs:", json.dumps([m[t] for m in input_manifest for t in ["raw","clean"]]))
(work/"diag_inputs_manifest.json").write_text(json.dumps(input_manifest,indent=2))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 3: attention-map probe\n"
            "Four initial inputs (raw/strip pairs from goal Camera and Noise), layers 0,10,12,23. "
            "Tool exposes action-to-video maps [24,32,98], not cross-text attention. "
            "Cannot directly attribute attention mass to suffix tokens with this interface. "
            "Will localize causally by cross-attention weight interventions.\n")
```
