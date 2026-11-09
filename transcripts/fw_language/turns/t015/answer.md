I will screen one alternative mechanism while waiting: unusually sharp text-attention heads in the action expert. This distinguishes a direct language-reading problem from contamination of the video representation and gives us a useful negative-control target for later probes.
```python
temperature_stats=[]
for r in head_stats:
    t=r["norm_q"]*r["norm_k"]
    for h in range(24):
        temperature_stats.append({
            "expert":r["expert"],"layer":r["layer"],"head":h,
            "temperature_proxy":float(t[h]),"relative":float(t[h]/np.median(t)),
            "q_bias_ratio":float(r["q_bias"][h]/max(r["q"][h],1e-10))
        })
print("Highest absolute cross-attention normalization products:")
for r in sorted(temperature_stats,key=lambda r:r["temperature_proxy"],reverse=True)[:12]:
    print("%s L%d H%d normQK=%.3f relative=%.3f query-bias/weight=%.4f" %
          (r["expert"],r["layer"],r["head"],r["temperature_proxy"],r["relative"],r["q_bias_ratio"]))
print("\nMost nonuniform within-layer cross-attention normalization products:")
for r in sorted(temperature_stats,key=lambda r:r["relative"],reverse=True)[:12]:
    print("%s L%d H%d normQK=%.3f relative=%.3f" %
          (r["expert"],r["layer"],r["head"],r["temperature_proxy"],r["relative"]))

query_offsets=[]
for expert in ("action","video"):
    for layer in range(30):
        p=f"mot.mixtures.{expert}.blocks.{layer}."
        q=load_original(p+"cross_attn.q.weight")
        qb=load_original(p+"cross_attn.q.bias")
        nb=load_original(p+"norm3.bias")
        ng=load_original(p+"norm3.weight")
        offset=(q@nb+qb).reshape(24,128).norm(dim=1)
        rms=(q*ng.unsqueeze(0)).reshape(24,128,-1).square().sum((1,2)).sqrt()
        for h in range(24):
            query_offsets.append((float(offset[h]/rms[h]),expert,layer,h,float(offset[h])))
print("\nLargest norm3-induced fixed query offsets / unit-input RMS:")
for r in sorted(query_offsets,reverse=True)[:12]:
    print("ratio=%.3f %s L%d H%d |offset|=%.3f" % r)
(W/"cross_attention_temperature_screen.json").write_text(json.dumps(temperature_stats))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate mechanisms to discriminate with fixed-input probes\n"
            "1. Direct action text-attention distraction / over-sharp attention.\n"
            "2. Paraphrase-sensitive video text writes, possibly residual channel 1938, "
            "then shared-attention reads into actions.\n"
            "3. Earlier text-embedding style sensitivity. Text matrices have no "
            "overwhelming single singular direction in static analysis.\n"
            "No mechanism is accepted until paired same-observation probes and rollouts.\n")
print("Weight-only screening completed; no candidate has behavioral evidence yet.")
```
