I will check for extreme Q/K normalization coordinates or unusually bias-dominated heads. Such features can create attention sinks, but I will only consider them causal candidates if the saved attention maps later show diversion toward texture-changed regions.

```python
sink_stats=[]
for l in range(30):
    for e,proj,d in [("action","q",1024),("video","k",3072)]:
        w=load_original(pname(e,l,f"self_attn.{proj}.weight")).reshape(24,128,d)
        b=load_original(pname(e,l,f"self_attn.{proj}.bias")).reshape(24,128)
        g=load_original(pname(e,l,f"self_attn.norm_{proj}.weight")).reshape(24,128)
        for h in range(24):
            sink_stats.append({"expert":e,"proj":proj,"layer":l,"head":h,
                               "gain_max":float(g[h].abs().max()),
                               "gain_rms":float(g[h].square().mean().sqrt()),
                               "gain_max_over_rms":float(g[h].abs().max()/g[h].square().mean().sqrt()),
                               "bias_over_weight_frob":float(b[h].norm()/w[h].norm()),
                               "max_gain_coordinate":int(g[h].abs().argmax())})
print("Largest gain concentration:")
for x in sorted(sink_stats,key=lambda x:x["gain_max_over_rms"],reverse=True)[:8]:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in x.items()})
print("Largest bias/weight ratios:")
for x in sorted(sink_stats,key=lambda x:x["bias_over_weight_frob"],reverse=True)[:8]:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in x.items()})
(W/"attention_sink_weight_reference.json").write_text(json.dumps(sink_stats,indent=2))
del w,b,g
```
