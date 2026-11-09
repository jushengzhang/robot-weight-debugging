All temporary edits have now been cleared. I’ll check which probe artifacts survived the restart and examine the previous band-readout construction. Reusing that verified measurement path should be less error-prone than building another one from scratch.
```python
print("WORK:", str(W), "active edits:",len(edited()))
print("TOP-LEVEL:", [p.name for p in W.iterdir()])
for base in [W/"fwd"/"roll_1", W/"fwd"/"roll_2", W/"replay_diag"]:
    print(str(base), [(p.name, "dir" if p.is_dir() else p.stat().st_size) for p in list(base.iterdir())[:6]])
print("SPLITS:", {k:len(v) for k,v in json.loads((W/"splits.json").read_text()).items()})
print("BAND PARAMETERS:")
bm=torch.load(W/"band_readout_metadata.pt",weights_only=False)
print({k:(v.tolist() if v.numel()<40 else [float(v.norm()),float(v.min()),float(v.max())]) if torch.is_tensor(v) else v for k,v in bm.items()})
snap=W/"fwd"/"probe_5_weights"
print("READOUT SNAPSHOT NAMES:", [p.stem for p in list(snap.iterdir())[:24]])
for stem in ["mot.mixtures.action.text_embedding.0.weight",
             "mot.mixtures.action.text_embedding.2.weight",
             "mot.mixtures.action.blocks.0.cross_attn.q.weight",
             "mot.mixtures.action.blocks.0.cross_attn.q.bias",
             "mot.mixtures.action.blocks.0.cross_attn.norm_q.weight",
             "mot.mixtures.action.blocks.0.cross_attn.k.weight",
             "mot.mixtures.action.blocks.0.cross_attn.norm_k.weight",
             "mot.mixtures.action.blocks.0.cross_attn.o.weight",
             "mot.mixtures.action.blocks.0.modulation"]:
    p=snap/(stem+".npy")
    if not p.exists(): continue
    a=torch.from_numpy(np.load(p))
    print(stem.split("action.")[1],tuple(a.shape),"nnz",a.count_nonzero().item(),
          "norm",round(a.norm().item(),3),
          "changed frac", round((a!=load_original(stem)).float().mean().item(),4))
    if "norm_" in stem or "q.bias" in stem:
        print("  nonzero items",[(int(i),float(a.flatten()[i])) for i in a.flatten().nonzero().flatten()[:12]])
print("PROBE DATA FILES (excluding weight snapshots):")
out=[]
for p in W.rglob("*"):
    if p.is_file() and ("probe_" in str(p)) and not any("_weights" in part for part in p.parts) and p.suffix in [".json",".npy",".npz"]:
        out.append(str(p.relative_to(W)))
print(out[:35], "count",len(out))
```
