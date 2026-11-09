The 27 high-/low-texture pairs have identical initial proprioception, instructions, and noise seeds, so they provide useful fixed-input contrasts. I will quantify the shifts by layer and select two pairs for attention-map inspection, rather than infer responsible heads from weight norms alone.
```python
pairs = json.loads((W/"diagnostic_pairs.json").read_text())
pairdiag = json.loads((W/"initial_pair_diagnostics.json").read_text())
paths0 = json.loads((W/"initial_probe_paths.json").read_text())
p3 = json.loads((W/"probe_3/results.json").read_text())
print("Probe metadata:", str(p3)[:2200])
with np.load(W/"probe_3/probe_orig_0_acts.npz") as z:
    print("key families:", sorted(set(k.split("::")[0] for k in z.files)))
    print("video key samples:", [k for k in z.files if "video" in k][:8])
print("Pair diagnostics:")
for d in sorted(pairdiag, key=lambda d:d["initial_chunk_rms_delta"], reverse=True)[:12]:
    print({k:d[k] for k in ["suite","high","low","hs","ls","initial_chunk_rms_delta","pixel_delta_cameras"]})
print("Layer 0 self-attention tensors:", [(n, SHAPES[n]) for n in NAMES
       if "action.blocks.0.self_attn" in n])
print("Layer 0 video attention tensors:", [(n, SHAPES[n]) for n in NAMES
       if "video.blocks.0.self_attn" in n])
print("Episode manifest size:", len(json.loads((W/"diagnostic_episodes.json").read_text())))
# Reload fixed-input summaries, retaining only returned mean activations.
A3 = []
for i in range(len(paths0)):
    with np.load(W/f"probe_3/probe_orig_{i}_acts.npz") as z:
        A3.append({k:z[k].astype(np.float32) for k in z.files if k.startswith("mean::")})
print("Mean activation dimensions:", [(k,v.shape) for k,v in list(A3[0].items())[:3]])
act3 = np.stack([np.load(W/f"probe_3/probe_orig_{i}_action.npy") for i in range(len(paths0))])
print("Predicted action array shape:", act3.shape)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 3: original fixed-input attention probe\n"
            "- 27 paired initial observations (54 inputs), same proprioception, instructions and seed.\n"
            "- Texture differences change action attention throughout the network. Early-layer shift/task ratios roughly 1.0-1.4; rank-4 contrast energy 0.65-0.85 in layers 0-5.\n"
            "- No causal inference yet; inspect head attention before selecting an edit. Passes used 3/40; no edits.\n")
```