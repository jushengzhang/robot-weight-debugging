I will choose a fixed, non-fitted combination of clean-context directions for the scan. I will also program a simple timestep-dependent selector offset; this changes only the temporary measurement circuit, not a repair candidate.
```python
g=torch.Generator().manual_seed(731)
coef=torch.randn(16,generator=g);coef[0]=0
u_scan=coef@Vhc[:16];u_scan-= (u_scan@Qprop)@Qprop.T;u_scan/=u_scan.norm()
ss=torch.einsum("nbd,d->nb",Xt,u_scan)
print("SCAN RANGE:",[round(float(z),3) for z in torch.quantile(ss.flatten(),torch.tensor([0.,.01,.1,.5,.9,.99,1.]))])
# The fixed scan range includes margins beyond measured band means.
scan_lo,scan_hi=-3.,3.
scan_centres=torch.linspace(scan_lo,scan_hi,30)
scan_knots=torch.linspace(scan_lo-.4,scan_hi+.4,96)
scan_step=float(scan_knots[1]-scan_knots[0])
def install_dense_readout(start,fresh=False):
    if fresh:clear_edits()
    eps=.001;nr=896;gdet=64.
    t0=load_original(A+"text_embedding.0.weight");b0=load_original(A+"text_embedding.0.bias")
    t2=load_original(A+"text_embedding.2.weight");b2=load_original(A+"text_embedding.2.bias")
    t0[:nr]=0;b0[:nr]=0
    ids=torch.arange(start,min(start+nr,4096))
    t0[torch.arange(len(ids)),ids]=eps
    t2[:nr]=0;t2[torch.arange(nr),torch.arange(nr)]=2/eps;b2[:nr]=0
    t0[896]=32*u_scan;t0[897]=-32*u_scan;b0[896:898]=0
    t2[896]=0;t2[896,896]=1/32;t2[896,897]=-1/32;b2[896]=0
    for j,c in enumerate(scan_knots):
        t0[898+j]=gdet*u_scan;b0[898+j]=-gdet*c
    t2[897]=0;t2[897,898:994]=scan_step/gdet;b2[897]=0
    for name,t in (("0.weight",t0),("0.bias",b0),("2.weight",t2),("2.bias",b2)):
        put(A+"text_embedding."+name,t)
    if not fresh:return
    # Timestep feature: exact GELU(z)-GELU(-z) readout of one existing sinusoidal coordinate.
    tw0=load_original(A+"time_embedding.0.weight");tb0=load_original(A+"time_embedding.0.bias")
    tw0[:2]=0;tw0[0,224]=1;tw0[1,224]=-1;tb0[:2]=0
    tw2=load_original(A+"time_embedding.2.weight");tb2=load_original(A+"time_embedding.2.bias")
    tw2[0]=0;tw2[0,0]=1;tw2[0,1]=-1;tb2[0]=0
    tpw=load_original(A+"time_projection.1.weight");tpb=load_original(A+"time_projection.1.bias")
    tpw[2*1024+1022]=0;tpw[2*1024+1022,0]=1;tpb[2*1024+1022]=0
    for part,t in (("time_embedding.0.weight",tw0),("time_embedding.0.bias",tb0),
                   ("time_embedding.2.weight",tw2),("time_embedding.2.bias",tb2),
                   ("time_projection.1.weight",tpw),("time_projection.1.bias",tpb)):
        put(A+part,t)
    aw=load_original(A+"action_encoder.weight");ab=load_original(A+"action_encoder.bias")
    aw*=1e-8;ab[:]=0;ab[1023]=500
    put(A+"action_encoder.weight",aw);put(A+"action_encoder.bias",ab)
    rows0=torch.arange(7)*128
    rows1=rows0+1
    qa=10000.;ka=1e8;tau=1800.
    offset_scale=float(scan_centres[1]-scan_centres[0])/.74
    for layer,c in enumerate(scan_centres):
        pre=A+f"blocks.{layer}."
        for part in ("self_attn.o.weight","ffn.2.weight"):
            t=load_original(pre+part);t[:]=0
            t[1020,0]=1e-8
            put(pre+part,t)
        for part in ("self_attn.o.bias","ffn.2.bias"):
            t=load_original(pre+part);t[:]=0;t[1020]=1e-8
            if layer==0 and part=="self_attn.o.bias":t[1022]=1
            put(pre+part,t)
        if layer==0:
            mod=load_original(pre+"modulation");mod[0,2,1022]=0;put(pre+"modulation",mod)
        n3w=load_original(pre+"norm3.weight");n3b=load_original(pre+"norm3.bias")
        n3w[1021:1023]=1;n3b[1021:1023]=0
        put(pre+"norm3.weight",n3w);put(pre+"norm3.bias",n3b)
        cp=pre+"cross_attn."
        qw=load_original(cp+"q.weight");qb=load_original(cp+"q.bias")
        qw[rows0]=0
        qw[rows0,1022]=offset_scale*500/math.sqrt(1024)
        qw[rows0,1021]=-offset_scale*500/math.sqrt(1024)
        qb[rows0]=float(c-scan_knots[0])-offset_scale*.47
        qw[rows1]=0;qb[rows1]=-1
        qw[-1]=0;qb[-1]=qa
        nq=load_original(cp+"norm_q.weight");nq[:]=0;nq[rows0]=qa/math.sqrt(3072);nq[rows1]=qa/math.sqrt(3072)
        kw=load_original(cp+"k.weight");kb=load_original(cp+"k.bias")
        kw[rows0]=0;kb[rows0]=0;kw[rows0,896]=tau*math.sqrt(128)
        kw[rows1]=0;kb[rows1]=0;kw[rows1,897]=tau*math.sqrt(128)
        kw[-1]=0;kb[-1]=ka
        nk=load_original(cp+"norm_k.weight");nk[:]=0;nk[rows0]=ka/math.sqrt(3072);nk[rows1]=ka/math.sqrt(3072)
        vw=load_original(cp+"v.weight");vb=load_original(cp+"v.bias")
        vw[:nr]=0;vw[torch.arange(nr),torch.arange(nr)]=1;vb[:nr]=0
        ow=load_original(cp+"o.weight");ob=load_original(cp+"o.bias")
        ow[:]=0;ow[torch.arange(nr),torch.arange(nr)]=1;ob[:]=0;ob[1020]=1e-8
        for part,t in (("q.weight",qw),("q.bias",qb),("norm_q.weight",nq),
                       ("k.weight",kw),("k.bias",kb),("norm_k.weight",nk),
                       ("v.weight",vw),("v.bias",vb),("o.weight",ow),("o.bias",ob)):
            put(cp+part,t)
install_dense_readout(0,True)
torch.save({"u":u_scan,"centres":scan_centres,"knots":scan_knots,"tau":1800.,"timestep_coord":224},W/"dense_readout_metadata.pt")
print("TEMPORARY DENSE READOUT:",len(edited()),"tensors")
```
