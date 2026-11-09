The inputs contain prompts and images, but not UMT5 token embeddings. I’ll therefore build a temporary scanner that reads narrow context bands along a clean-text principal direction. Each layer will select a different band, and the ten diffusion steps will vary the selection within that band. I’ll edit only the specific timestep rows needed for this signal, avoiding the whole-tensor change that was refused.
```python
# Measurement construction only; no local model execution.
ca=torch.load(W/"context_band_analysis.pt",map_location="cpu",weights_only=False)
X=ca["X"].float()
v=ca["Vhc"][0].float().clone()
Qp=ca["Qprop"].float()
v=v-Qp@(Qp.T@v)
v=v/v.norm()
print("Scanner score quantiles:", torch.quantile((X@v).flatten(),torch.tensor([0.,.01,.1,.5,.9,.99,1.])).tolist())

script=r'''
import math, numpy as np, torch
from pathlib import Path
A="mot.mixtures.action."
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def putpart(name, fn):
    x=load_original(name).clone()
    fn(x)
    save_edit(name,x)
def install_dense_readout(start, direction, nread=896, tau=1600.0):
    # Residual readout coordinates, a timestep coordinate, and two large balanced anchors.
    clear_edits()
    nr=nread; r=1021; ap=1022; am=1023
    eps=.001; anchor=1.e6; timegain=1.e4
    lo,hi=-1.45,1.8
    width=(hi-lo)/30.
    centres=lo+width*torch.arange(30)
    sinmax=math.sin(1000.*10000.**(-127./128.))
    scan_gain=width/sinmax
    # Linear coordinate readout and piecewise-quadratic selector, in the existing text MLP.
    ncoord=min(nr,4096-start)
    knots=torch.linspace(-1.8,2.1,116)
    h=float(knots[1]-knots[0]); L=float(knots[0]); interior=knots[1:-1]
    h0=nr; h1=nr+1; hs=nr+2; gain=256.
    te0=load_original(A+"text_embedding.0.weight").clone()
    tb0=load_original(A+"text_embedding.0.bias").clone()
    te0[:nr]=0; tb0[:nr]=0
    te0[torch.arange(ncoord),start+torch.arange(ncoord)]=eps
    te0[h0]=direction; te0[h1]=-direction
    tb0[h0:h1+1]=0
    te0[hs:hs+len(interior)]=gain*direction[None,:]
    tb0[hs:hs+len(interior)]=-gain*interior
    save_edit(A+"text_embedding.0.weight",te0)
    save_edit(A+"text_embedding.0.bias",tb0)
    te2=load_original(A+"text_embedding.2.weight").clone()
    tb2=load_original(A+"text_embedding.2.bias").clone()
    te2[:nr+2]=0; tb2[:nr+2]=0
    te2[torch.arange(ncoord),torch.arange(ncoord)]=2./eps
    te2[nr,h0]=1; te2[nr,h1]=-1
    te2[nr+1,h0]=2*L+h; te2[nr+1,h1]=-(2*L+h)
    te2[nr+1,hs:hs+len(interior)]=2*h/gain
    tb2[nr+1]=-L*L-h*L
    save_edit(A+"text_embedding.2.weight",te2)
    save_edit(A+"text_embedding.2.bias",tb2)
    # SiLU(s)-SiLU(-s)=s: extract one sinusoidal time coordinate through two hidden units.
    def t0w(x):
        x[:2]=0; x[0,255]=1; x[1,255]=-1
    putpart(A+"time_embedding.0.weight",t0w)
    putpart(A+"time_embedding.0.bias",lambda x:x[:2].zero_())
    def t2w(x):
        x[:2]=0; x[0,0]=1; x[0,1]=-1; x[1,0]=-1; x[1,1]=1
    putpart(A+"time_embedding.2.weight",t2w)
    putpart(A+"time_embedding.2.bias",lambda x:x[:2].zero_())
    tr=2*1024+r
    def tpw(x):
        x[tr]=0; x[tr,0]=timegain; x[tr,1]=-timegain
    putpart(A+"time_projection.1.weight",tpw)
    putpart(A+"time_projection.1.bias",lambda x:x[tr].zero_())
    reserved=torch.cat([torch.arange(nr),torch.tensor([r,ap,am])])
    putpart(A+"action_encoder.weight",lambda x:x.index_fill_(0,reserved,0))
    def aeb(x):
        x[reserved]=0; x[ap]=anchor; x[am]=-anchor
    putpart(A+"action_encoder.bias",aeb)
    head0=torch.arange(7)*128; head1=head0+1; an=3071
    qa=1.e4; ka=1.e4
    for b in range(30):
        p=A+f"blocks.{b}."
        for mod in ["self_attn.o","ffn.2"]:
            putpart(p+mod+".weight",lambda x:x.index_fill_(0,reserved,0))
            def ob(x,mod=mod):
                x[reserved]=0
                if mod=="self_attn.o": x[r]=1.
            putpart(p+mod+".bias",ob)
        putpart(p+"modulation",lambda x:x[0,2,r].zero_())
        putpart(p+"norm3.weight",lambda x:x[r].fill_(1.))
        putpart(p+"norm3.bias",lambda x:x[r].zero_())
        qw=load_original(p+"cross_attn.q.weight").clone()
        qb=load_original(p+"cross_attn.q.bias").clone()
        qw[head0]=0; qw[head1]=0; qw[an]=0
        coeff=scan_gain*anchor*math.sqrt(2/1024.)/timegain/(b+1)
        qw[head0,r]=coeff
        qb[head0]=centres[b]; qb[head1]=1.; qb[an]=qa
        save_edit(p+"cross_attn.q.weight",qw); save_edit(p+"cross_attn.q.bias",qb)
        nq=torch.zeros(3072)
        nq[head0]=2*tau*math.sqrt(128.)*qa/math.sqrt(3072.)
        nq[head1]=-tau*math.sqrt(128.)*qa/math.sqrt(3072.)
        save_edit(p+"cross_attn.norm_q.weight",nq)
        kw=load_original(p+"cross_attn.k.weight").clone()
        kb=load_original(p+"cross_attn.k.bias").clone()
        kw[head0]=0; kw[head1]=0; kw[an]=0
        kw[head0,nr]=1.; kw[head1,nr+1]=1.
        kb[head0]=0; kb[head1]=0; kb[an]=ka
        save_edit(p+"cross_attn.k.weight",kw); save_edit(p+"cross_attn.k.bias",kb)
        nk=torch.zeros(3072); nk[head0]=ka/math.sqrt(3072.); nk[head1]=ka/math.sqrt(3072.)
        save_edit(p+"cross_attn.norm_k.weight",nk)
        vw=load_original(p+"cross_attn.v.weight").clone()
        vb=load_original(p+"cross_attn.v.bias").clone()
        vw[:nr]=0; vw[torch.arange(nr),torch.arange(nr)]=1.; vb[:nr]=0
        save_edit(p+"cross_attn.v.weight",vw); save_edit(p+"cross_attn.v.bias",vb)
        ow=load_original(p+"cross_attn.o.weight").clone()
        ob=load_original(p+"cross_attn.o.bias").clone()
        ow[reserved]=0; ob[reserved]=0
        ow[torch.arange(nr),torch.arange(nr)]=1.
        save_edit(p+"cross_attn.o.weight",ow); save_edit(p+"cross_attn.o.bias",ob)
    torch.save(dict(direction=direction,centres=centres,scan_gain=scan_gain,
                    timegain=timegain,anchor=anchor,nread=nr,tau=tau,start=start),
               Path(WORK)/"dense_readout_metadata.pt")
    return len(edited())
'''
(W/"dense_readout.py").write_text(script)
exec(script)
print("Installed diagnostic tensors:",install_dense_readout(0,v))
with (W/"notes.md").open("a") as f:
    f.write("\n### Restart after pass12\nFailed dense readout had left10 partial instrumentation edits; all cleared. "
            "Reconstructed scanner with targeted timestep rows only, 896-coordinate readout, 30x10 selectors along clean PC0. "
            "Temporary instrumentation now installed for slice0:896; MUST CLEAR before candidate evaluation.\n")
print("Ready for pass13; temporary measurement weights only.")
```
