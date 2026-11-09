I will solve the residual identity—layer increment equals gated attention plus gated MLP—to estimate the time-only gates. This is activation analysis, not weight fitting. I will check the reconstruction error before using it to identify which pathway introduces the loop-associated signal.
```python
for n in list(edited()): drop_edit(n)
meta21=json.loads((W/"probe_21"/"results.json").read_text())
assert [probe_identity(r) for r in meta21]==ident19
acts21=[np.load(localpath(r["activations"])) for r in meta21]
T21=np.stack([np.stack([z[f"calls::expert.{l}.attn_out::action"] for l in range(18)]) for z in acts21])
ML=M18[late18].astype(np.float64)
AT=T21.astype(np.float64)
DR=(R19[:,1:]-R19[:,:-1]).astype(np.float64)
aa=(AT[:,1:]**2).sum(0);mm=(ML[:,1:]**2).sum(0)
am=(AT[:,1:]*ML[:,1:]).sum(0)
ad=(AT[:,1:]*DR).sum(0);md=(ML[:,1:]*DR).sum(0)
det=aa*mm-am*am
ga=(ad*mm-md*am)/np.maximum(det,1e-15)
gm=(md*aa-ad*am)/np.maximum(det,1e-15)
ERR=DR-ga[None]*AT[:,1:]-gm[None]*ML[:,1:]
print("Gate reconstruction relative RMS",
      np.round(np.sqrt((ERR**2).sum((0,2,3))/(DR**2).sum((0,2,3))),4).tolist())
PA=ga[None]*AT[:,1:];PM=gm[None]*ML[:,1:]
grA=PA@gn;grM=PM@gn
print("L | gated attention grip F/S | gated MLP grip F/S | MLP norm F/S | MLP open-only effect")
for l in range(1,18):
    a=grA[:,l-1].mean(1);m=grM[:,l-1].mean(1)
    nm=np.linalg.norm(PM[:,l-1],axis=-1).mean(1)
    print(l,np.round([a[~succ19].mean(),a[succ19].mean(),
                      m[~succ19].mean(),m[succ19].mean(),
                      nm[~succ19].mean(),nm[succ19].mean(),
                      es(m,~succ19&open19,succ19&open19)],3).tolist())
print("Gate mean/std, per-time cosine to first call")
for l in [1,7,10,14,16,17]:
    for k,G in [("A",ga[l-1]),("M",gm[l-1])]:
        cs=(G@G[0])/(np.linalg.norm(G,axis=1)*np.linalg.norm(G[0])+1e-12)
        print(l,k,np.round([G.mean(),G.std(),np.min(cs),np.median(cs)],3).tolist())
np.savez(W/"gate_reconstruction.npz",attention=ga.astype("float32"),
         mlp=gm.astype("float32"),relative_error=np.sqrt((ERR**2).sum((0,2,3))/(DR**2).sum((0,2,3))))
with (W/"notes.md").open("a") as f:
    f.write("\n## Camera-key rollout rejected, pass20\n"
            "K14 camera-axis beta=.5:26/48->26/48,2 rescues/2 regressions,loops21->18. Reset.\n"
            "\n## Gate-aware attribution, pass21\n"
            "Collected matched attention outputs to complement MLP and residual probes; "
            "infer time-only gates from residual identity, not by executing model locally or training weights.\n")
```
