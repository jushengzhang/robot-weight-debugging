The image library is unavailable, but the saved policy inputs already contain the exact image tensors. I will use those directly for the matched-image checks.

```python
def first_input_path(suite,vid,roll=2):
    return W/f"roll_{roll}_frames"/f"{suite}_v{vid}_s17_original"/"t0000_inputs.pt"

initial_inputs={}
for r in roll2:
    p=first_input_path(r["suite"],r["id"])
    initial_inputs[(r["suite"],r["id"])]=torch.load(p,map_location="cpu",weights_only=False)
pixel_masks={}
pair_diagnostics=[]
for p in pairs_meta:
    if p["low"] is None: continue
    kh=(p["suite"],p["high"]); kl=(p["suite"],p["low"])
    a=initial_inputs[kh]; b=initial_inputs[kl]
    im1=a["input_image"].float().numpy().squeeze(0).transpose(1,2,0)
    im2=b["input_image"].float().numpy().squeeze(0).transpose(1,2,0)
    di=np.abs(im1-im2).mean(-1)
    ac1=np.asarray(a["action_chunk"]); ac2=np.asarray(b["action_chunk"])
    pair_diagnostics.append({
        "suite":p["suite"],"high":p["high"],"low":p["low"],
        "hs":r2map[kh]["success"],"ls":r2map[kl]["success"],
        "proprio_delta":float((a["proprio"]-b["proprio"]).norm()),
        "prompt_equal":a["prompt"]==b["prompt"],
        "noise_seeds":[a["seed"],b["seed"]],
        "image_scale":[float(im1.min()),float(im1.max())],
        "pixel_delta_cameras":[float(di[:,:224].mean()),float(di[:,224:].mean())],
        "initial_chunk_rms_delta":float(np.sqrt(np.mean((ac1-ac2)**2))),
    })
    pixel_masks[kh]=di
print("Pair checks (suite, IDs, success, proprio delta, prompt match, seeds, image deltas, action delta):")
for p in pair_diagnostics:
    print(p["suite"],p["high"],p["low"],p["hs"],p["ls"],round(p["proprio_delta"],5),
          p["prompt_equal"],p["noise_seeds"],np.round(p["pixel_delta_cameras"],4).tolist(),
          round(p["initial_chunk_rms_delta"],4))
print("Input image range:",pair_diagnostics[0]["image_scale"])
print("Matched failure/success pairs:",
      [(p["suite"],p["high"],p["low"]) for p in pair_diagnostics if not p["hs"] and p["ls"]])
probe_pairs=[p for p in pairs_meta if p["low"] is not None]
probe_paths=[str(first_input_path(p["suite"],p[k])) for p in probe_pairs for k in ["high","low"]]
(W/"initial_probe_paths.json").write_text(json.dumps(probe_paths))
(W/"initial_pair_diagnostics.json").write_text(json.dumps(pair_diagnostics,indent=2))
print("Probe paths:",len(probe_paths))
print("Pairs grouped by suite:",
      {s:[(p["high"],p["low"]) for p in probe_pairs if p["suite"]==s]
       for s in sorted(set(v["suite"] for v in variants))})
```
