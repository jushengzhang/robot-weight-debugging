The exceptional video modes share almost the same output direction, unlike the corresponding action heads. I will check whether this direction is merely removed by layer normalization and map its strongest routes into action attention. This guards against mistaking a harmless residual offset for a behavioral mechanism.
```python
U0 = U.clone()
ref = U0[0]
U0 *= torch.sign(U0@ref).unsqueeze(1)
_, su, vut = torch.linalg.svd(U0, full_matrices=False)
common_u = vut[0]
print("Common output direction: first-PC energy", round(float(su[0]**2/su.square().sum()),4))
print("|cos with all-ones|", round(float(common_u.sum().abs()/math.sqrt(common_u.numel())),5))
vals,inds = common_u.abs().topk(10)
print("Top absolute coordinates:",[(int(i),round(float(common_u[i]),5)) for i in inds])
print("Inverse participation dimension:",round(float(1/common_u.pow(4).sum()),1))
print("\nLayer-normalization weights on common direction:")
for l in (0,13,15,17,20,22,23,24,27,29):
    g=load_original(f"mot.mixtures.video.blocks.{l}.norm3.weight")
    b=load_original(f"mot.mixtures.video.blocks.{l}.norm3.bias")
    print(l,"|gamma*u|",round(float((g*common_u).norm()),4),
          "bias projection",round(float(b@common_u),4))

read_routes=[]
for src in (13,15,17,20,22,23,24,27):
    tgt=src+1
    pv=f"mot.mixtures.video.blocks.{tgt}.self_attn."
    pa=f"mot.mixtures.action.blocks.{tgt}.self_attn."
    VV=load_original(pv+"v.weight").reshape(24,128,-1)
    KK=load_original(pv+"k.weight").reshape(24,128,-1)
    AO=load_original(pa+"o.weight").reshape(-1,24,128).permute(1,0,2)
    # Weight-space transfer, normalized by random-direction RMS sensitivity.
    vu=VV@common_u
    transfer=torch.einsum("hij,hj->hi",AO,vu).norm(dim=1)
    gram_o=AO.transpose(1,2)@AO
    gram_v=VV@VV.transpose(1,2)
    random_rms=(torch.einsum("hij,hji->h",gram_o,gram_v)/3072).clamp_min(1e-12).sqrt()
    kn=(KK@common_u).norm(dim=1)/(KK.square().sum((1,2))/3072).sqrt()
    ratio=transfer/random_rms
    for h in range(24):
        read_routes.append({"src":src,"layer":tgt,"head":h,"value_gain":float(ratio[h]),
                            "key_gain":float(kn[h]),"transfer":float(transfer[h])})
print("\nStrongest shared-attention read routes relative to random-direction RMS:")
for r in sorted(read_routes,key=lambda r:r["value_gain"],reverse=True)[:18]:
    print("video L%d -> action L%d H%d: value=%.2fx key=%.2fx abs=%.3f" %
          (r["src"],r["layer"],r["head"],r["value_gain"],r["key_gain"],r["transfer"]))
torch.save(common_u,W/"video_common_mode.pt")
(W/"common_mode_read_routes.json").write_text(json.dumps(read_routes))
print("Current edits:",edited())
```
