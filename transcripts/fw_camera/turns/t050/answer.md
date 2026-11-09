The replay changes the diagnosis: attenuating the layer-29 heads barely affects camera-pair action disagreement. Layers 11 and 16 instead show large, coherent shifts—one direction explains 62% and 74% of their camera-pair activation differences. I will check whether those directions recur across independent task subsets before editing them.
```python
direction_stats = {}
for l in (11,16,29):
    x = torch.from_numpy(D[:,l,0])
    _,s,vh = torch.linalg.svd(x,full_matrices=False)
    u = vh[0]
    _,_,va = torch.linalg.svd(x[::2],full_matrices=False)
    _,_,vb = torch.linalg.svd(x[1::2],full_matrices=False)
    reference = torch.from_numpy(AG[:,2,l,0])
    mild = torch.from_numpy(AG[:,0,l,0])
    severe = torch.from_numpy(AG[:,1,l,0])
    co = [t @ u for t in (mild,severe,reference)]
    vals = {
        "split_direction_cos": float(abs(va[0]@vb[0])),
        "heldhalf_energy_fraction": float((x[1::2]@va[0]).square().sum()/x[1::2].square().sum()),
        "pair_direction_rms": float((x@u).square().mean().sqrt()),
        "light_coefficient_mean": float(co[2].mean()),
        "light_coefficient_sd": float(co[2].std()),
        "mild_coefficient_mean": float(co[0].mean()),
        "severe_coefficient_mean": float(co[1].mean()),
        "mild_coefficient_sd": float(co[0].std()),
        "severe_coefficient_sd": float(co[1].std()),
    }
    direction_stats[l] = vals
    print("L",l,{k:round(v,4) for k,v in vals.items()})
print("L11/L16 direction cosine:", round(float(bases[11][0]@bases[16][0]),4))
(W/"camera_direction_stability.json").write_text(json.dumps(direction_stats,indent=2))
with notes.open("a") as f:
    f.write("\n## Passes 6-7: L29 output hypothesis weak\n"
            "- L29 H4/7/8/17 half-output: +3.1 pp, SE 3.1 on same 128 discovery episodes (10 gains, 6 losses).\n"
            "- Same-state XYZ discrepancies remain 0.08288 / 0.08585 vs original 0.08301 / 0.08552; first4 action changes only ~0.001 XYZ RMS.\n"
            "- Therefore L29 agent-locked heads are not established as principal causal failure mechanism; no best promoted.\n"
            "- Original 19 paired first-call action attention contrasts: L11 RMS .619, rank1 energy .616; L16 RMS .616, rank1 energy .742. L29 rank1 only .214.\n"
            "- Analyze stable camera-shift directions, then test reversible low-rank output projection with a preserved mild-light mean.\n"
            "- Forward passes used 7/40.\n")
```
