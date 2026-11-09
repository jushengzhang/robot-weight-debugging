While the backend is offline, I will check whether the high-gain heads concentrate their effect in a narrow subspace. If they do, a low-rank correction could preserve useful language information better than suppressing an entire head. I will also reserve practice variants for validation before examining any outcomes.
```python
import hashlib

def split_key(v):
    s = f"{v['suite']}:{v['id']}:language_repair_split_v1"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10

partitions = {
    "discovery": [v for v in variants if split_key(v) < 4],
    "validation": [v for v in variants if 4 <= split_key(v) < 7],
    "confirmation": [v for v in variants if split_key(v) >= 7],
}
(W / "practice_partitions.json").write_text(json.dumps(partitions))
for group, vs in partitions.items():
    print(group, len(vs), "language", sum(v["category"]=="Language Instructions" for v in vs))

def ov_spectrum(expert, layer, head, vectors=False):
    p = f"mot.mixtures.{expert}.blocks.{layer}.cross_attn."
    O = load_original(p+"o.weight")[:,head*128:(head+1)*128]
    V = load_original(p+"v.weight")[head*128:(head+1)*128,:]
    Qo, Ro = torch.linalg.qr(O, mode="reduced")
    Qv, Rv = torch.linalg.qr(V.T, mode="reduced")
    U, S, VT = torch.linalg.svd(Ro @ Rv.T, full_matrices=False)
    if vectors:
        return S, Qo@U, VT@Qv.T
    return S

t0 = time.time()
spectral = []
for expert in ("action", "video"):
    for layer, heads in ((13, range(24)), (15, [0,4,8,14,20]), (9, [0,12,14,23])):
        for head in heads:
            s = ov_spectrum(expert, layer, head)
            spectral.append({
                "expert":expert, "layer":layer, "head":head, "s1":float(s[0]),
                "s2":float(s[1]), "s1_frac":float(s[0]**2/(s.square().sum())),
                "effective_rank":float(s.square().sum()**2/s.pow(4).sum())
            })
print("\nLargest singular gain of effective OV maps:")
for r in sorted(spectral, key=lambda x:x["s1"], reverse=True)[:18]:
    print("%s L%d H%d s1=%.3f s2=%.3f rank1_energy=%.1f%% effrank=%.2f" %
          (r["expert"],r["layer"],r["head"],r["s1"],r["s2"],100*r["s1_frac"],r["effective_rank"]))
(W/"ov_spectral_screen.json").write_text(json.dumps(spectral, indent=2))
print("Elapsed seconds:", round(time.time()-t0,1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Weight-only screening (not causal evidence)\n"
            "- Cross-attention L13 H14 has largest within-layer V/O Frobenius gain: "
            "2.81x median in action, 2.59x in video. L15 H4 next (1.94x/1.81x).\n"
            "- Final self-attention norm outlier channel 606 is shared by both experts; "
            "not assumed to be a language defect.\n"
            "- Practice variants assigned deterministically to discovery (40%), "
            "validation (30%), confirmation (30%); no outcomes seen yet. "
            "Initial random baseline, if run, is diagnostic only.\n")
```
