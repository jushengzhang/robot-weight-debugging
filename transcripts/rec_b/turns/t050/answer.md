I will measure how much each late layer actually changes the residual stream, and compare failures with controls issuing the same gripper command. This is a check against editing a large but normal MLP output or a signal that merely encodes “open.”
```python
meta19=json.loads((W/"probe_19"/"results.json").read_text())
ident19=[probe_identity(r) for r in meta19]
succ19=np.array([bidx[t[:3]]["success"] for t in ident19])
a19=np.stack([np.load(localpath(r["action"])) for r in meta19])
acts19=[np.load(localpath(r["activations"])) for r in meta19]
R19=np.stack([np.stack([z[f"calls::expert.{l}.resid::action"] for l in range(18)]) for z in acts19])
print("Residual shape",R19.shape)
pR=R19@gn;nR=np.linalg.norm(R19,axis=-1)
open19=a19[:,:,6].mean(1)<0
def es(v, f, s):
    return (v[f].mean()-v[s].mean())/np.sqrt((v[f].var()+v[s].var())/2+1e-10)
print("L | residual norm F/S | increment norm F/S | grip residual F/S | grip effect(all/open-only)")
for l in range(18):
    rn=nR[:,l].mean(1);gp=pR[:,l].mean(1)
    dn=np.linalg.norm(R19[:,l]-R19[:,max(0,l-1)],axis=-1).mean(1)
    vals=[rn[~succ19].mean(),rn[succ19].mean(),dn[~succ19].mean(),dn[succ19].mean(),
          gp[~succ19].mean(),gp[succ19].mean(),es(gp,~succ19,succ19),
          es(gp,~succ19&open19,succ19&open19)]
    print(l,np.round(vals,3).tolist())
print("Adaptive normalization bias/gate weight magnitudes:")
for l in [0,3,7,10,14,16,17]:
    for tag in ["input_layernorm","post_attention_layernorm"]:
        bn=f"{E}{l}.{tag}.dense.bias";wn=f"{E}{l}.{tag}.dense.weight"
        b=load_original(bn).reshape(3,1024);t=load_original(wn).reshape(3,1024,1024)
        print(l,tag.split("_")[0],"bias mean/std",
              np.round(b.mean(1).numpy(),3).tolist(),np.round(b.std(1).numpy(),3).tolist(),
              "weight RMS",np.round(t.square().mean((1,2)).sqrt().numpy(),4).tolist())
# Static action-direction selectivity of late attention heads.
U=torch.linalg.qr(Aout[:6].T,mode="reduced").Q
for l in [5,7,14,16,17]:
    o=load_original(f"{E}{l}.self_attn.o_proj.weight")
    print("O",l,"norm/head",np.round(o.reshape(1024,8,256).norm(dim=(0,2)).numpy(),2).tolist(),
          "grip/head",np.round((g@o).reshape(8,256).norm(dim=1).numpy(),3).tolist())
print("Final residual grip projection by call F/S:",
      np.round(pR[~succ19,17].mean(0),3).tolist(),np.round(pR[succ19,17].mean(0),3).tolist())
print("MLP effects conditioned on open actions (late)")
p18late=p18[late18];n18late=norm18[late18]
# Probe 18 late entries are ordered exactly as probe 19.
assert [t for t in ident18 if t[3]>0]==ident19
print([(l,round(float(es(p18late[:,l].mean(1),~succ19&open19,succ19&open19)),2))
       for l in range(18)])
```
