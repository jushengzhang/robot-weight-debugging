The independent test did not establish an overall benefit: six rescues were offset by seven losses, and loop failures increased. I will not retain the broad query softening. Since its L2 gains repeated while other levels suffered, I will check whether one dominant query mode—not the whole head—accounts for the excessive instruction fixation.
```python
for n in list(edited()): drop_edit(n)
qfull=load_original(qname)
spectra={}
for h in range(8):
    U,S,Vh=torch.linalg.svd(qfull[256*h:256*(h+1)],full_matrices=False)
    spectra[h]=(U,S,Vh)
    print("L16H",h,"top sv",np.round(S[:5].numpy(),3).tolist(),
          "top1/top4 energy",round(float(S[0]**2/S.square().sum()),3),
          round(float(S[:4].square().sum()/S.square().sum()),3))
nw=load_original(f"{expert}layers.16.input_layernorm.dense.weight")
nb=load_original(f"{expert}layers.16.input_layernorm.dense.bias")
mod16=cond@nw.T+nb
qobs=[]
for rr0 in j4:
    z=np.load(rr0["activations"])
    rr=torch.from_numpy(z["calls::expert.15.resid::action"]).float()
    xn=rr/(rr.square().mean(-1,keepdim=True)+1e-6).sqrt()
    xn=xn*(1+mod16[:,:1024])+mod16[:,1024:2048]
    qobs.append(xn@qfull[768:1024].T)
qobs=torch.stack(qobs)
U,S,Vh=spectra[3]
print("Four diagnostic input alignments with leading query modes:")
for rr0 in j5[:4]:
    match=next((i for i,rr in enumerate(j4) if pathlib.Path(rr["input"]).resolve()==pathlib.Path(rr0["input"]).resolve()),None)
    if match is not None:
        qq=qobs[match,0]
        print(match,rr0["input"].split("/")[-1],
              "q norm",round(qq.norm().item(),3),
              "top6 q projections",np.round((qq@U[:,:6]).numpy(),3).tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 35: independent E16H3 query-softening test rejected\n"
            "- 122 new paired-start episodes, seed 29: 64/122 -> 63/122, -0.8 pp (SE 3.0), 6 rescues / 7 losses.\n"
            "- Loop failures 40 -> 43. L2 +9.4 pp repeated the discovery direction, but overall no gain and solved-scene damage. Reverted.\n"
            "- Independent panel 1 now consumed; panels 2-4 remain untouched.\n")
(W/"current_candidate.json").write_text(json.dumps({"name":"none"}))
```
