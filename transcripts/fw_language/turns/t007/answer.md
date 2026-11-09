The spectral check changed the leading hypothesis: layer 13, video head 7 has a single mode carrying 76.5% of its effective text-to-residual energy, whereas head 14's large gain is spread across many modes. I will check how exceptional that rank-one mode is across layers and whether the following shared-attention layer can carry it into actions.
```python
t0 = time.time()
print("Layer 13 comparison:")
for expert in ("action","video"):
    rows = [r for r in spectral if r["expert"]==expert and r["layer"]==13]
    for r in sorted(rows, key=lambda r:r["s1"], reverse=True)[:5]:
        print(expert, "head", r["head"], "s1", round(r["s1"],3), "s2",round(r["s2"],3),
              "rank1_energy",round(r["s1_frac"],3))
# Load matrices once per layer, compute each head's exact low-rank OV spectrum.
all_spectral = []
mode_cache = {}
for expert in ("action","video"):
    for layer in range(30):
        p = f"mot.mixtures.{expert}.blocks.{layer}.cross_attn."
        Oall = load_original(p+"o.weight")
        Vall = load_original(p+"v.weight")
        rows = []
        for head in range(24):
            O = Oall[:, head*128:(head+1)*128]
            V = Vall[head*128:(head+1)*128, :]
            Qo, Ro = torch.linalg.qr(O, mode="reduced")
            Qv, Rv = torch.linalg.qr(V.T, mode="reduced")
            U,S,VT = torch.linalg.svd(Ro @ Rv.T, full_matrices=False)
            rows.append({"expert":expert,"layer":layer,"head":head,
                         "s1":float(S[0]), "s2":float(S[1]),
                         "frac":float(S[0]**2/S.square().sum())})
            if expert=="video" and layer==13 and head==7:
                mode_cache["video_13_7"] = {
                    "s":S.clone(), "u":(Qo@U[:,0]).clone(),
                    "v":(VT[0]@Qv.T).clone(), "o_head":O.clone(), "v_head":V.clone()
                }
        med = np.median([r["s1"] for r in rows])
        for r in rows:
            r["relative_s1"] = r["s1"]/med
        all_spectral.extend(rows)
print("\nTop relative singular gains across all layers:")
for r in sorted(all_spectral, key=lambda r:r["relative_s1"], reverse=True)[:16]:
    print("%s L%d H%d s1=%.3f s2=%.3f rel=%.2f rank1_energy=%.1f%%" %
          (r["expert"],r["layer"],r["head"],r["s1"],r["s2"],r["relative_s1"],100*r["frac"]))
u = mode_cache["video_13_7"]["u"]
print("\nNext-layer shared-attention sensitivity to video L13 H7 output mode:")
read_sensitivity = []
for layer in (14,15):
    pv = f"mot.mixtures.video.blocks.{layer}.self_attn."
    pa = f"mot.mixtures.action.blocks.{layer}.self_attn."
    vk = load_original(pv+"k.weight").reshape(24,128,-1) @ u
    vv = load_original(pv+"v.weight").reshape(24,128,-1) @ u
    ao = load_original(pa+"o.weight").reshape(-1,24,128).permute(1,0,2)
    val_transfer = torch.einsum("hij,hj->hi",ao,vv).norm(dim=1)
    for head in range(24):
        read_sensitivity.append((float(val_transfer[head]), layer, head,
                                 float(vk[head].norm()),float(vv[head].norm())))
for r in sorted(read_sensitivity,reverse=True)[:10]:
    print("action-value transfer %.3f L%d H%d |K u|=%.3f |V u|=%.3f" % r)
torch.save(mode_cache["video_13_7"],W/"video_L13_H7_OV_mode.pt")
(W/"all_cross_attention_spectra.json").write_text(json.dumps(all_spectral))
print("Elapsed seconds:",round(time.time()-t0,1))
```
