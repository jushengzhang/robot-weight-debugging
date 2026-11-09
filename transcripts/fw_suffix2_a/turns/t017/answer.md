The old detector responds to about 28% of context tokens for camera, initial-state, and noise tags, but a threshold of 1 suppresses only part of that mass and misses almost all light, background, and layout tags. I will now read context vectors in narrow score bands. Unlike a single context mean, these measurements can distinguish added tag-like tokens from shifted task-token representations.
```python
centres=torch.tensor([-.4,-.3,-.2,-.15,-.1,-.05,0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8])
NR=896
def install_band_readout(start, fresh=False):
    if fresh:
        clear_edits()
    eps=.001
    t0=load_original(A+"text_embedding.0.weight"); b0=load_original(A+"text_embedding.0.bias")
    t2=load_original(A+"text_embedding.2.weight"); b2=load_original(A+"text_embedding.2.bias")
    # Near-linear coordinate readout: 2*GELU(eps*x)/eps, with explicit error checks across bands later.
    t0[:NR]=0; b0[:NR]=0
    ids=torch.arange(start,min(start+NR,4096))
    t0[torch.arange(len(ids)),ids]=eps
    t2[:NR]=0; b2[:NR]=0
    t2[torch.arange(NR),torch.arange(NR)]=2/eps
    # Exact linear u and narrow GELU hinges for attention band selection.
    t0[896]=32*u_prior; t0[897]=-32*u_prior
    b0[896:898]=0
    t2[896]=0; t2[896,896]=1/32; t2[896,897]=-1/32; b2[896]=0
    for j,c in enumerate(centres):
        t0[898+j]=64*u_prior; b0[898+j]=-64*c
        t2[897+j]=0; t2[897+j,898+j]=1/64; b2[897+j]=0
    for name,t in (("0.weight",t0),("0.bias",b0),("2.weight",t2),("2.bias",b2)):
        put(A+"text_embedding."+name,t)
    if not fresh:
        return
    for part in ("action_encoder.weight","action_encoder.bias"):
        t=load_original(A+part);t[:NR]=0;put(A+part,t)
    score_rows=torch.arange(7)*128
    qa=10000.;ka=1000000.;tau=100.
    for layer in range(30):
        pre=A+f"blocks.{layer}."
        for part in ("self_attn.o.weight","self_attn.o.bias","ffn.2.weight","ffn.2.bias"):
            t=load_original(pre+part);t[:NR]=0;put(pre+part,t)
        cp=pre+"cross_attn."
        qw=load_original(cp+"q.weight");qb=load_original(cp+"q.bias")
        qw[score_rows]=0;qb[score_rows]=1
        qw[-1]=0;qb[-1]=qa
        nq=load_original(cp+"norm_q.weight");nq[:]=0;nq[score_rows]=qa/math.sqrt(3072)
        kw=load_original(cp+"k.weight");kb=load_original(cp+"k.bias")
        kw[score_rows]=0;kb[score_rows]=0
        kw[score_rows,896]=tau*math.sqrt(128)
        kw[score_rows,897+layer]=-2*tau*math.sqrt(128)
        kw[-1]=0;kb[-1]=ka
        nk=load_original(cp+"norm_k.weight");nk[:]=0;nk[score_rows]=ka/math.sqrt(3072)
        vw=load_original(cp+"v.weight");vb=load_original(cp+"v.bias")
        vw[:NR]=0;vw[torch.arange(NR),torch.arange(NR)]=1;vb[:NR]=0
        ow=load_original(cp+"o.weight");ob=load_original(cp+"o.bias")
        ow[:NR]=0;ow[torch.arange(NR),torch.arange(NR)]=1;ob[:NR]=0
        for part,t in (("q.weight",qw),("q.bias",qb),("norm_q.weight",nq),
                       ("k.weight",kw),("k.bias",kb),("norm_k.weight",nk),
                       ("v.weight",vw),("v.bias",vb),("o.weight",ow),("o.bias",ob)):
            put(cp+part,t)
install_band_readout(0,fresh=True)
torch.save({"centres":centres,"u":u_prior,"readout_dim":NR,"eps":.001,"tau":100},W/"band_readout_metadata.pt")
print("TEMPORARY BAND READOUT:",len(edited()),"tensors; slice0:896")
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass4 detector histogram\nAt u>.5 raw tail mass~.156 vs clean/paraphrase~0. Camera/initial/noise~.281; background/light~.042; layout~.011. At u>1 raw~.089 and almost no background/light/layout. Old detector incompletely covers tags. Histogram removed.\nTemporary 30-band context readout now installed (action only): near-linear896 coordinate slices, exact u and GELU hinges; constant anchored q/k select bands at30 centres. Five slices planned. MUST CLEAR before candidate.\n")
```
