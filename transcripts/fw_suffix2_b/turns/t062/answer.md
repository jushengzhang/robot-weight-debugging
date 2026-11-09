The sampled clean and paraphrased contexts stay below detector score 0.45, while many tagged samples exceed 0.5. I will combine the rank-256 correction with a sharper 0.5-threshold suffix gate, preserving the detector’s view of the original context so the projection cannot erase its signal. This is a test candidate, not a retained edit.
```python
def apply_hybrid(rank=256,dose=1.,threshold=.5,gain=128.,strength=6.):
    apply_context_candidate(rank,dose)
    projected={n:load(n).clone() for n in edited()}
    for p in (W/"prior_candidate").glob("*.npy"):
        n=p.stem
        a=torch.from_numpy(np.load(p).copy())
        if n in projected:
            d=(a-load_original(n)).abs().sum(dim=1)
            rows=torch.where(d>0)[0]
            q=projected[n].clone()
            q[rows]=a[rows]*(gain/32.)
            a=q
        elif n.endswith("text_embedding.0.bias"):
            rows=torch.where(a!=load_original(n))[0]
            a[rows]=-gain*threshold
        elif n.endswith("text_embedding.2.weight"):
            ch=702 if ".action." in n else 1066
            h=296 if ".action." in n else 2844
            a[ch,h]=1./gain
        elif n.endswith("cross_attn.norm_q.weight") and strength!=6.:
            o=load_original(n)
            changed=a!=o
            a[changed]*=strength/6.
        save_edit(n,a)
    cfg={"kind":"hybrid_oblique_suffix_gate","rank":rank,"dose":dose,
         "threshold":threshold,"gain":gain,"strength":strength,"experts":["action","video"]}
    (W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
    print("hybrid tensors",len(edited()))
    return cfg
apply_hybrid()
with (W/"notes.md").open("a") as f:
    f.write("\nPass24 candidate: combine rank256 oblique both-expert projection with prior single-detector suffix q/k gate, threshold.5, gain128 (sharper than prior32), strength6. Gate reads ORIGINAL input via dedicated neuron restored after projection, so correction cannot hide suffix detector signal. All measured clean/paraphrase query scores <=.449/.413 vs tagged max2.548. 308 tensors; NOT final. Static analytic composition, no training.\n")
```
