The simulator is still offline, so no pass was charged. I will use this time to measure head-wise video-to-action coupling and Q/K normalization gains. These measurements can identify candidates to inspect later, but weight magnitudes alone are not evidence that a head causes background failures.

```python
t0 = time.time()
head_stats = []
def pname(expert, layer, suffix):
    return f"mot.mixtures.{expert}.blocks.{layer}.{suffix}"

for l in range(30):
    av = load_original(pname("action", l, "self_attn.v.weight")).reshape(24,128,1024)
    vv = load_original(pname("video", l, "self_attn.v.weight")).reshape(24,128,3072)
    ao = load_original(pname("action", l, "self_attn.o.weight")).reshape(1024,24,128).permute(1,0,2).contiguous()
    gram_o = torch.bmm(ao.transpose(1,2), ao)
    gram_v = torch.bmm(vv, vv.transpose(1,2))
    gram_av = torch.bmm(av, av.transpose(1,2))
    coupling = (gram_o * gram_v).sum((1,2)).clamp_min(0).sqrt()
    self_coupling = (gram_o * gram_av).sum((1,2)).clamp_min(0).sqrt()
    aqg = load_original(pname("action", l, "self_attn.norm_q.weight")).reshape(24,128)
    vkg = load_original(pname("video", l, "self_attn.norm_k.weight")).reshape(24,128)
    vkb = load_original(pname("video", l, "self_attn.k.bias")).reshape(24,128)
    vvb = load_original(pname("video", l, "self_attn.v.bias")).reshape(24,128)
    q = load_original(pname("action", l, "self_attn.q.weight")).reshape(24,128,1024)
    k = load_original(pname("video", l, "self_attn.k.weight")).reshape(24,128,3072)
    for h in range(24):
        head_stats.append({
            "layer":l, "head":h, "video_coupling":float(coupling[h]),
            "action_coupling":float(self_coupling[h]),
            "q_gain_rms":float(aqg[h].square().mean().sqrt()),
            "k_gain_rms":float(vkg[h].square().mean().sqrt()),
            "q_norm":float(q[h].norm()), "k_norm":float(k[h].norm()),
            "v_norm":float(vv[h].norm()), "o_norm":float(ao[h].norm()),
            "k_bias_norm":float(vkb[h].norm()), "v_bias_norm":float(vvb[h].norm()),
        })
np.savez(W/"weight_head_stats.npz", **{k:np.array([x[k] for x in head_stats]) for k in head_stats[0]})
print("Seconds:", round(time.time()-t0,1))
print("Per-layer coupling max/median, strongest head, Qgain max, Kgain max")
for l in range(30):
    ss = head_stats[24*l:24*(l+1)]
    cs = np.array([x["video_coupling"] for x in ss])
    print(l, round(cs.max()/np.median(cs),2), int(cs.argmax()),
          round(max(x["q_gain_rms"] for x in ss),3),
          round(max(x["k_gain_rms"] for x in ss),3))
print("Largest video/action coupling ratios:")
for x in sorted(head_stats, key=lambda x:x["video_coupling"]/max(x["action_coupling"],1e-9), reverse=True)[:12]:
    print({k:round(v,3) if isinstance(v,float) else v for k,v in x.items()})
del av,vv,ao,gram_o,gram_v,gram_av,q,k
with (W/"notes.md").open("a") as f:
    f.write("\n## Setup\n"
            "- Tool offline at first request; 0 passes charged.\n"
            "- Practice catalogue: 524 background variants; fixed SHA256 split gives 163 diagnosis / 361 validation.\n"
            "- 34 matched BG/light task pairs, seed 17, prepared in diagnostic_episodes.json.\n"
            "- Tensor convention: mot.mixtures.{video,action}.blocks.L; 24 heads x 128 channels.\n"
            "- Weight-only head coupling statistics saved; these are candidate-ranking information, not causal evidence.\n")
```
