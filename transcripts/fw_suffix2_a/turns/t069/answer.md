The new category detectors have a useful measured margin: clean scores stay below 0.017 and matched-prefix scores below 0.086, while the median maximum score on unmatched tagged features is about 0.35. I’ll set a threshold of 0.20 and reuse the previous attention-gate wiring, changing the detector rather than the gate mechanism.
```python
def install_dense_gate(threshold=.20, dose=1., experts=("action","video"), gain=128.):
    clear_edits()
    olddir=W/"fwd"/"probe_11_weights"
    specs={}
    # Reuse established targeted q/k/value-column wiring; no self-attention, MLP or action-head edits.
    for ex in experts:
        pfx=f"mot.mixtures.{ex}."
        ch=int(gm["specs"][ex]["channel"])
        ow=load_original(pfx+"text_embedding.2.weight")
        hidden=torch.argsort(ow.square().sum(0))[:len(dirs)]
        w0=load_original(pfx+"text_embedding.0.weight").clone()
        b0=load_original(pfx+"text_embedding.0.bias").clone()
        w2=ow.clone();b2=load_original(pfx+"text_embedding.2.bias").clone()
        w0[hidden]=gain*dirs; b0[hidden]=-gain*threshold
        w2[:,hidden]=0;w2[ch]=0;w2[ch,hidden]=1./gain;b2[ch]=0
        for suf,t in [("text_embedding.0.weight",w0),("text_embedding.0.bias",b0),
                      ("text_embedding.2.weight",w2),("text_embedding.2.bias",b2)]:
            save_edit(pfx+suf,t)
        for fp in olddir.iterdir():
            name=fp.name[:-4]
            if not name.startswith(pfx+"blocks."):continue
            x=torch.from_numpy(np.load(fp))
            if name.endswith("cross_attn.q.bias") and dose!=1.:
                org=load_original(name)
                changed=x!=org
                x[changed]*=dose
            save_edit(name,x)
        specs[ex]=dict(hidden=hidden,channel=ch)
    torch.save(dict(directions=dirs,threshold=threshold,dose=dose,gain=gain,specs=specs),
               W/"dense_gate_metadata.pt")
    return len(edited())
print("Installed six-detector gate tensors:",install_dense_gate())
print("Detector overlap matrix diagonal/offdiagonal max:",
      float((dirs@dirs.T).diag().mean()),float((dirs@dirs.T-torch.eye(6)).abs().max()))
with (W/"notes.md").open("a") as f:
    f.write("\n### Candidate dense unmatched-context gate\nSix category-mean novel-feature directions; protect512 clean PCs+256 raw-prefix PCs+proprio. "
            "Threshold.20,gain128,prior probe11 q/k gate wiring strength12,key64,both experts. "
            "Measured max clean detector.017, max matched-prefix.086, unmatched median max.349. Probe20 pending.\n")
```
