The available activations are token-averaged, so the earlier mean detector could not show which individual context tokens it suppressed. I will temporarily turn part of action block 0 into a readout of that detector’s score distribution. This is measurement-only instrumentation, not a candidate repair, and it will be removed before evaluating behavior.
```python
A="mot.mixtures.action."
V="mot.mixtures.video."
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def put(n,t):
    save_edit(n,t.contiguous().float())
def reserve_block0_readout(nread=1023):
    # Zero writes only in reserved residual coordinates.
    for part in ("action_encoder.weight","action_encoder.bias",
                 "blocks.0.self_attn.o.weight","blocks.0.self_attn.o.bias",
                 "blocks.0.ffn.2.weight","blocks.0.ffn.2.bias"):
        t=load_original(A+part); t[:nread]=0; put(A+part,t)
    nq=load_original(A+"blocks.0.cross_attn.norm_q.weight")
    nq[:1024]=0
    put(A+"blocks.0.cross_attn.norm_q.weight",nq)
    vv=load_original(A+"blocks.0.cross_attn.v.weight")
    vv[:1024]=0
    vv[torch.arange(nread),torch.arange(nread)]=1
    vb=load_original(A+"blocks.0.cross_attn.v.bias"); vb[:1024]=0
    oo=load_original(A+"blocks.0.cross_attn.o.weight")
    oo[:nread]=0
    oo[torch.arange(nread),torch.arange(nread)]=1
    ob=load_original(A+"blocks.0.cross_attn.o.bias"); ob[:nread]=0
    for part,t in (("v.weight",vv),("v.bias",vb),("o.weight",oo),("o.bias",ob)):
        put(A+"blocks.0.cross_attn."+part,t)
def install_histogram():
    clear_edits()
    nr=1023
    thresholds=torch.linspace(-4,12,nr)
    g=64.
    t0=load_original(A+"text_embedding.0.weight")
    b0=load_original(A+"text_embedding.0.bias")
    t2=load_original(A+"text_embedding.2.weight")
    b2=load_original(A+"text_embedding.2.bias")
    t0[:nr]=g*u_prior
    b0[:nr]=-g*thresholds
    t2[:nr]=0
    t2[torch.arange(nr),torch.arange(nr)]=1/g
    b2[:nr]=0
    for name,t in (("0.weight",t0),("0.bias",b0),("2.weight",t2),("2.bias",b2)):
        put(A+"text_embedding."+name,t)
    reserve_block0_readout(nr)
    return thresholds
hist_thresholds=install_histogram()
print("TEMPORARY INSTRUMENTATION:",len(edited()),"tensors")
base_actions=np.stack([np.load(r["action"]) for r in rec3])
base_norm=np.stack([np.load(r["action_norm"]) for r in rec3])
print("BASE ACTION RMS GAP:",float(np.sqrt(np.mean((base_actions[:24]-base_actions[24:48])**2))))
print("BASE NORMALIZED ACTION RMS GAP:",float(np.sqrt(np.mean((base_norm[:24]-base_norm[24:48])**2))))
for layer in (0,1,3,7,12,18,23,29):
    xs=np.stack([np.load(r["activations"])[f"calls::action.{layer}.resid::action"][0] for r in rec3])
    print("layer",layer,"first-step contrast RMS",round(float(np.sqrt(np.mean((xs[:24]-xs[24:48])**2))),5))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass3 reference and temporary instrumentation\n56 inputs:24 raw/clean pairs+8 paraphrases; noise_seed0. Probe exposes action residual means, not token embeddings/cross-text maps. Temporary block0 uniform readout measures old detector score hinge moments at1023 thresholds [-4,12]. MUST REMOVE instrumentation before repair evaluation.\n")
```
