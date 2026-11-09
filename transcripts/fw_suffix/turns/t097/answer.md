The broader gate kept clean drift small but worsened the overall tagged-action match, especially for backgrounds. The lower-threshold gate also failed earlier. I’ll therefore test a more selective single-detector gate: raise the threshold from 0.5 to 1.0 while keeping its other settings fixed.
```python
def apply_single_gate(threshold=0.5, strength=6.0, gain=32.0,
                      experts=("action","video")):
    for n in list(edited()): drop_edit(n)
    for f in sorted((W/"fwd/probe_25_weights").glob("*.npy")):
        n=f.name[:-4]
        ex="action" if ".action." in n else "video"
        if ex not in experts: continue
        x=torch.from_numpy(np.load(f)).float()
        de=gate["designs"][ex]
        h,j=de["hidden"],de["channel"]
        if n.endswith("text_embedding.0.weight"):
            x[h]=gain*gate["u"]
        elif n.endswith("text_embedding.0.bias"):
            x[h]=-gain*threshold
        elif n.endswith("text_embedding.2.weight"):
            x[j,h]=1.0/gain
        elif n.endswith("cross_attn.norm_q.weight"):
            layer=int(n.split(".blocks.")[1].split(".")[0])
            co=de["layers"][layer]["coords"]
            x[co]*=strength/6.0
        save_edit(n,x)
    meta={"name":"single_selective_context_attention_gate",
          "threshold":threshold,"strength":strength,"detector_gain":gain,
          "key_gain":64.0,"experts":list(experts),"edited_count":len(edited())}
    (W/"current_edit.json").write_text(json.dumps(meta,indent=2))
    return meta
meta=apply_single_gate(threshold=1.0,strength=6.0,gain=32.0)
(W/"candidate_single_threshold1.json").write_text(json.dumps(meta,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Four-detector gate rejected after pass32\n"
            "Tag action gap ratio1.0224; background ratio1.8667. Clean/language drift "
            ".00136/.00147, but no overall tag improvement. Restored the original "
            "single-detector architecture, changed only threshold.5->1.0 "
            "(strength6,gain32,both experts), for an isolated selectivity test.\n")
print(json.dumps(meta,indent=2))
```
