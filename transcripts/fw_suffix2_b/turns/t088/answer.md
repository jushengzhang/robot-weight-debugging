The old gate leaves many unmatched samples below threshold. Six protected, category-derived directions separate these samples better: their scores are at most 0.004 on the measured clean/paraphrased contexts, and at most 0.289 on clearly task-like raw samples. I’ll add thresholded detectors at 0.35 to the existing gate, keeping the rank-256 projection and attention-gate strength fixed.
```python
update_hybrid_projection(rank=256, dose=1.0)
gate_meta = {"directions": new_dirs.float(), "categories": catlist,
             "threshold": .35, "gain": 128., "hidden": {}}
for ex, old_h, out_ch in (("action",296,702), ("video",2844,1066)):
    pre = f"mot.mixtures.{ex}.text_embedding."
    w0o = load_original(pre+"0.weight")
    w2o = load_original(pre+"2.weight")
    costs = w0o.norm(dim=1) * w2o.norm(dim=0)
    hs = [int(i) for i in torch.argsort(costs) if int(i) != old_h][:6]
    a = load(pre+"0.weight")
    b = load(pre+"0.bias")
    c = load(pre+"2.weight")
    print(ex, "new hidden coordinates", hs,
          "old gate coefficient", float(c[out_ch, old_h]),
          "removed column energy fraction",
          float(w2o[:,hs].square().sum()/w2o.square().sum()))
    for h, v in zip(hs, new_dirs.float()):
        a[h] = 128. * v
        b[h] = -128. * .35
        c[:,h] = 0.
        c[out_ch,h] = 1./128.
    save_edit(pre+"0.weight", a)
    save_edit(pre+"0.bias", b)
    save_edit(pre+"2.weight", c)
    gate_meta["hidden"][ex] = hs
torch.save(gate_meta, W/"expanded_gate_meta.pt")
cfg = {"kind":"hybrid_oblique_expanded_suffix_gate",
       "basis_file":"rich_context_projection.pt","rank":256,"dose":1.0,
       "old_threshold":.5,"new_threshold":.35,"gain":128.,"strength":6.,
       "detectors":7,"experts":["action","video"]}
(W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
snapshot_candidate("hybrid_r256_expanded_gate")
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 planned expanded suffix gate: rank256 projection restored. "
            "Keep original detector; add6 static normalized category-mean unmatched-context "
            "directions after removing521D protected span. Threshold.35 exceeds "
            "max score.2894 on clearly task-like (nearest-clean cosine>.8) raw samples; "
            "measured clean/paraphrase score max<.004. "
            "Six additional low-weight-energy hidden neurons per expert feed the SAME "
            "gate output channel and existing q/k gate; gain128, strength6 unchanged. "
            "No training, no action-target fitting. Test by paired development rollout.\n")
print("Candidate installed:", cfg, "edited tensors:", len(edited()))
```
