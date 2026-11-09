The corrected contrast separates empty-hand observations from controls on unseen scenes, but imperfectly: pairwise separation is 82.9%. I will check its scores on the four original diagnostic observations and estimate how selectively it could reduce the instruction-focused query component before committing to an edit.
```python
Qn=E+"layers.16.self_attn.q_proj.weight"
Q0=load_original(Qn)
mode=torch.load(W/"E16H3_top_query_mode.pt",weights_only=False)
u=mode["left"].float()
qmode=Xq@Q0[768:1024].T@u
print("Dominant query-mode means by physical group (calls 0,4,9):")
for g in ["fully_closed","holding_like","open"]:
    print(g,torch.round(qmode[groups==g].mean(0)[[0,4,9]]*100)/100)
for j,(idx,scene,grip,step) in enumerate([(2,367,"open",200),(3,367,"close",100),(6,753,"open",200),(7,753,"close",50)]):
    xx=torch.from_numpy(np.load(rr4[idx]["activations"])["calls::expert.15.resid::action"]).float()
    xq=xx/(xx.square().mean(-1,keepdim=True)+1e-6).sqrt()*(1+scale)+shift
    ss1=xq@d/norm_scale
    obs=[p for p in cat if p["id"]==scene and p["grip"]==grip and p["step"]==step]
    width=abs(obs[0]["q1"]-obs[0]["q2"]) if obs else None
    print("Diagnostic",j,"width",round(width,4) if width is not None else None,
          "cue scores",np.round(ss1.numpy()[[0,4,9]],3).tolist(),
          "query mode",np.round((xq@Q0[768:1024].T@u).numpy()[[0,4,9]],2).tolist())
amp=float(qmode[(groups=="fully_closed")&derive].mean())/norm_scale
delta=-0.5*amp*torch.outer(u,d)
print("Proposed rank-one relative change to head:",float(delta.norm()/Q0[768:1024].norm()),
      "amp",amp)
```
