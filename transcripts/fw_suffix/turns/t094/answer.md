The output-space projection failed: tagged chunks moved farther from the clean reference, and clean drift increased. I’ll discard it. Next I’ll test a selective attention gate using the previously measured background, lighting, and layout detectors alongside the common detector, with thresholds above their observed clean/paraphrase responses.
```python
for n in list(edited()): drop_edit(n)

# Multi-detector gate: static directions/thresholds from pass23, no fitting.
multi_spec=[(0,0.5),(5,0.625),(6,0.875),(7,0.625)]
multi_design={}
for ex in ("action","video"):
    p=f"mot.mixtures.{ex}.text_embedding."
    w0=load_original(p+"0.weight")
    w2=load_original(p+"2.weight")
    h0=gate["designs"][ex]["hidden"]
    score=w2.square().sum(0)*w0.square().sum(1)
    order=torch.argsort(score).tolist()
    hs=[h0]+[h for h in order if h!=h0][:len(multi_spec)-1]
    multi_design[ex]={"hidden":hs,"channel":gate["designs"][ex]["channel"]}
    print(ex,"reserved hidden neurons",hs,
          "column norm quantiles",torch.quantile(w2.norm(dim=0),torch.tensor([0.,.5,1.])).tolist())
print("Detector orthogonality to proprio:",
      [float((Pctx.T@gd["directions"][i]).abs().max()) for i,t in multi_spec])

def apply_multi_gate(strength=12.0, key_gain=64.0, gain=128.0, spec=multi_spec):
    for n in list(edited()): drop_edit(n)
    # Restore the established single-coordinate q/k/v gate architecture.
    root=W/"fwd/probe_25_weights"
    for f in sorted(root.glob("*.npy")):
        n=f.name[:-4]
        if ".text_embedding." in n: continue
        x=torch.from_numpy(np.load(f)).float()
        ex="action" if ".action." in n else "video"
        if n.endswith("cross_attn.norm_q.weight"):
            layer=int(n.split(".blocks.")[1].split(".")[0])
            co=gate["designs"][ex]["layers"][layer]["coords"]
            x[co]*=strength/6.0
        if n.endswith("cross_attn.k.weight") and key_gain!=64.0:
            layer=int(n.split(".blocks.")[1].split(".")[0])
            co=gate["designs"][ex]["layers"][layer]["coords"]
            j=gate["designs"][ex]["channel"]
            x[co,j]*=key_gain/64.0
        save_edit(n,x)
    for ex in ("action","video"):
        pre=f"mot.mixtures.{ex}.text_embedding."
        hs=multi_design[ex]["hidden"][:len(spec)]
        j=multi_design[ex]["channel"]
        a=load_original(pre+"0.weight")
        b=load_original(pre+"0.bias")
        c=load_original(pre+"2.weight")
        d=load_original(pre+"2.bias")
        c[:,hs]=0.; c[j]=0.; d[j]=0.
        for h,(di,t) in zip(hs,spec):
            a[h]=gain*gd["directions"][di]
            b[h]=-gain*t
            c[j,h]=1./gain
        for suf,x in (("0.weight",a),("0.bias",b),("2.weight",c),("2.bias",d)):
            save_edit(pre+suf,x)
    meta={"name":"multi_selective_context_attention_gate","strength":strength,
          "key_gain":key_gain,"detector_gain":gain,"detectors":spec,
          "design":multi_design,"edited_count":len(edited())}
    (W/"current_edit.json").write_text(json.dumps(meta,indent=2))
    return meta

meta=apply_multi_gate(strength=12.0)
torch.save({"design":multi_design,"spec":multi_spec},W/"multi_gate_design.pt")
(W/"candidate_multi_gate.json").write_text(json.dumps(meta,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Prefix output projector rejected\n"
            "Pass31 tag gap ratio1.1037, clean drift.02956, language drift.00960. "
            "Both edited tensors removed. Next candidate: four selective detector gates "
            "(common threshold.5, background.625, light.875, layout.625), gain128, "
            "attention strength12, both experts. Built from pass23 readout without fitting; "
            "no prefix projector combined.\n")
print(json.dumps(meta,indent=2))
```
