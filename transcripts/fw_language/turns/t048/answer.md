Layer 13 head 14 has unusually large value and output gains in both experts, but no exceptional query/key temperature. That makes amplification of instruction features a more plausible target than attention softening. I will examine its effective value-to-residual modes and compare them with the observed residual changes before intervening.
```python
torch.set_num_threads(min(12, os.cpu_count() or 12))
spec=json.loads((W/"all_cross_attention_spectra.json").read_text())
for e in ["action","video"]:
    print("\n", e, "L13 spectra")
    for x in spec:
        if x["expert"]==e and x["layer"]==13 and x["head"] in [7,14,18,19]:
            print(x)
    print("Top relative singular values, excluding video sink-dominated heads:")
    rows=[x for x in spec if x["expert"]==e]
    print(compact_obj(sorted(rows,key=lambda x:x["relative_s1"],reverse=True)[:7],1800))

# Effective OV SVD using thin QR: analysis of weights only.
def ov_modes(e,l,h,nmode=6):
    pfx=f"mot.mixtures.{e}.blocks.{l}.cross_attn."
    O=load_original(pfx+"o.weight")[:,128*h:128*(h+1)].contiguous()
    V=load_original(pfx+"v.weight")[128*h:128*(h+1)].contiguous()
    qo,ro=torch.linalg.qr(O, mode="reduced")
    qv,rv=torch.linalg.qr(V.T, mode="reduced")
    u,s,vh=torch.linalg.svd(ro@rv.T,full_matrices=False)
    return {"U":(qo@u[:,:nmode]).contiguous(),"s":s,
            "R":(qv@vh[:nmode].T).contiguous(),"O":O,"V":V}
modes={}
for e in ["action","video"]:
    for l,h in [(13,14),(15,4),(11,19)]:
        m=ov_modes(e,l,h)
        modes[(e,l,h)]=m
        print(e,l,h,"singular",np.round(m["s"][:6].numpy(),3).tolist(),
              "top-energy",round(float(m["s"][0]**2/(m["s"]**2).sum()),4))
        if e=="action":
            d=torch.from_numpy(D[:,l]-(D[:,l-1] if l else 0))
            U=m["U"]
            frac=(d@U).square().sum()/(d.square().sum()+1e-9)
            print(" residual difference increment energy in first 6 OV modes:",round(frac.item(),4))
torch.save({str(k):{a:b for a,b in m.items() if a in ["U","s","R"]}
            for k,m in modes.items()}, W/"localized_cross_modes.pt")
print("\nInput list for all natural pairs:")
print(json.dumps([r["input"] for r in pr4]))
```
