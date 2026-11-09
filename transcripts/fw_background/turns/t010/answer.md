The backend is still unavailable. I will inspect FFN channel norms next, because a texture disturbance could be amplified by a few MLP channels before an action head reads it. I will keep this as a weight-only reference and make no edits until the simulator provides evidence.

```python
t0 = time.time()
ffn_summary = []
ffn_scores = {}
for expert, d in [("video",3072),("action",1024)]:
    for l in range(30):
        w1 = load_original(pname(expert,l,"ffn.0.weight"))
        b1 = load_original(pname(expert,l,"ffn.0.bias"))
        w2 = load_original(pname(expert,l,"ffn.2.weight"))
        incoming = w1.norm(dim=1)
        outgoing = w2.norm(dim=0)
        product = incoming*outgoing
        order = torch.argsort(product,descending=True)
        ffn_scores[(expert,l)] = {
            "incoming":incoming.numpy(), "outgoing":outgoing.numpy(),
            "product":product.numpy(), "bias":b1.numpy(),
        }
        ffn_summary.append({
            "expert":expert,"layer":l,
            "max_to_median":float(product.max()/product.median()),
            "top8_energy":float(product[order[:8]].square().sum()/product.square().sum()),
            "top_channels":order[:5].tolist(),
        })
print("FFN scan seconds:", round(time.time()-t0,1))
for e in ["video","action"]:
    ss = [x for x in ffn_summary if x["expert"]==e]
    print(e, "max/median range:", [round(min(x["max_to_median"] for x in ss),2),
                                    round(max(x["max_to_median"] for x in ss),2)])
    print("Most concentrated layers:")
    for x in sorted(ss,key=lambda x:x["top8_energy"],reverse=True)[:6]:
        print({k:round(v,5) if isinstance(v,float) else v for k,v in x.items()})
np.savez(W/"ffn_weight_statistics.npz",
         **{f"{e}_{l}_{k}":a for (e,l),ss in ffn_scores.items() for k,a in ss.items()})
(W/"weight_analysis_summary.json").write_text(json.dumps(
    {"head_stats":head_stats,"ffn_summary":ffn_summary},indent=2))
del w1,w2,b1
print("No weight edits:", len(edited()))
```
