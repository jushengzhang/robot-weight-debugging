The five-detector gate worsened numeric-tag actions sharply, despite little clean drift. That suggests it is suppressing useful context as well as suffix information, so I will remove it. Next I will test a different intervention: projecting text input onto the measured clean-language span, preserving task-like components rather than dropping whole context tokens.
```python
def apply_clean_manifold(rank=256,dose=1.,experts=("action","video")):
    clear_edits()
    Q=torch.linalg.qr(torch.cat([Qprop,Vhc[:rank].T],1),mode="reduced").Q
    for expert in experts:
        n=f"mot.mixtures.{expert}.text_embedding.0.weight"
        w=load_original(n)
        pw=(w@Q)@Q.T
        nw=w+dose*(pw-w)
        put(n,nw)
        print(expert,"relative changed norm",round(float((nw-w).norm()/w.norm()),4))
    return Q
Qman=apply_clean_manifold(256,1.)
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass11 five-prototype gate rejected\nGap ratio1.5522; clean drift.00229, language.00138. Numeric categories worsened (Camera2.55x); background improved. Indicates low-score novelty is not safe to mask wholesale.\nNext: both-expert clean-language manifold projection, rank256 clean PCs+exact proprio span (2 text entry tensors). Removes input innovations while retaining their task-like components, unlike token gating. Probe12 pending.\n")
```
