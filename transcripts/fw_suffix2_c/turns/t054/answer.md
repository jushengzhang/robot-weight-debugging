I’ll now collect weighted context summaries rather than only uniform averages. Temporary readout weights will emphasize different parts of the text context using fixed linear directions derived from existing weights and activation statistics. These are diagnostic edits only; I’ll remove them before testing another repair.
```python
m17=compare_screen(17)
snapshot("candidate_video_union_half")
clear_edits()
torch.set_num_threads(min(16, os.cpu_count() or 16))

# Fixed, data-derived linear readout directions; no fitting to action targets.
v0=load_original("mot.mixtures.video.text_embedding.0.weight").double()
vb=load_original("mot.mixtures.video.text_embedding.0.bias").double()
v2=load_original("mot.mixtures.video.text_embedding.2.weight").double()
pr=torch.cat([load_original("proprio_encoder.weight").double(),
              load_original("proprio_encoder.bias").double()[:,None]],dim=1)
Qp=torch.linalg.qr(pr,mode="reduced").Q
u_tag=torch.from_numpy(np.load(W/"fwd/probe_10_weights/mot.mixtures.action.text_embedding.0.weight.npy"))[-1].double()
hc=H[48:,0]-H[48:,0].mean(0)
_,ss,Vh=torch.linalg.svd(hc,full_matrices=False)
ds=torch.stack([u_tag, v0.T@Vh[0], v0.T@Vh[1]],dim=1)
ds=ds-Qp@(Qp.T@ds)
ds=ds/ds.norm(dim=0,keepdim=True)
print("READOUT DIRECTION COSINES", (ds.T@ds).round(decimals=4).tolist())

m=1018
excluded=torch.argsort(v2.square().sum(0))[:18]
keep=torch.tensor([i for i in range(3072) if i not in set(excluded.tolist())])
query_spec=[(-1,0.0)]+[(j,g) for j in range(3) for g in [-16.,-4.,4.,16.]]
torch.save({"directions":ds,"feature_ids":keep,"excluded":excluded,"queries":query_spec,"m":m},
           W/"weighted_readout_spec.pt")

instrument_code = '''
import numpy as np, torch
from pathlib import Path
W=Path(WORK)
def clear_edits():
    for n in list(edited()): drop_edit(n)
def install_weighted_readout(slice_id, expert="video"):
    clear_edits()
    spec=torch.load(W/"weighted_readout_spec.pt",weights_only=False)
    m=spec["m"]; dirs=spec["directions"]
    if expert=="video":
        ids=spec["feature_ids"][slice_id*m:(slice_id+1)*m]
    else:
        w=load_original("mot.mixtures.action.text_embedding.2.weight")
        ids=torch.argsort(w.square().sum(0),descending=True)[:m].sort().values
    p="mot.mixtures.action."
    src=f"mot.mixtures.{expert}.text_embedding."
    w0=load_original(src+"0.weight"); b0=load_original(src+"0.bias")
    w=torch.zeros_like(load_original(p+"text_embedding.0.weight"))
    b=torch.zeros_like(load_original(p+"text_embedding.0.bias"))
    w[:m]=w0[ids]; b[:m]=b0[ids]
    for j in range(3):
        w[m+2*j]=dirs[:,j].float()
        w[m+2*j+1]=-dirs[:,j].float()
    save_edit(p+"text_embedding.0.weight",w)
    save_edit(p+"text_embedding.0.bias",b)
    z=torch.zeros_like(load_original(p+"text_embedding.2.weight"))
    z[:m,:m]=torch.eye(m)
    for j in range(3):
        z[m+j,m+2*j]=1
        z[m+j,m+2*j+1]=-1
    save_edit(p+"text_embedding.2.weight",z)
    b=load_original(p+"text_embedding.2.bias")
    b[:1023]=0
    save_edit(p+"text_embedding.2.bias",b)
    for name in ["action_encoder.weight","action_encoder.bias"]:
        a=load_original(p+name); a[:1023]=0; save_edit(p+name,a)
    anchors=torch.arange(24)*128
    for l,(direction,gain) in enumerate(spec["queries"]):
        bp=p+f"blocks.{l}."
        for name in ["self_attn.o.weight","self_attn.o.bias","ffn.2.weight","ffn.2.bias"]:
            a=load_original(bp+name); a[:1023]=0; save_edit(bp+name,a)
        cp=bp+"cross_attn."
        a=load_original(cp+"q.weight"); a[anchors]=0; save_edit(cp+"q.weight",a)
        a=load_original(cp+"q.bias"); a[anchors]=1000.; save_edit(cp+"q.bias",a)
        a=torch.zeros_like(load_original(cp+"norm_q.weight"))
        a[anchors]=gain if direction>=0 else 1.
        save_edit(cp+"norm_q.weight",a)
        a=torch.zeros_like(load_original(cp+"k.weight"))
        if direction>=0:
            a[anchors,m+direction]=1.
        else:
            a[-1]=load_original(cp+"k.weight")[-1]
        save_edit(cp+"k.weight",a)
        a=torch.ones_like(load_original(cp+"k.bias")); a[anchors]=0; save_edit(cp+"k.bias",a)
        save_edit(cp+"norm_k.weight",torch.ones_like(load_original(cp+"norm_k.weight")))
        a=torch.zeros_like(load_original(cp+"v.weight"))
        a[:1024,:1024]=torch.eye(1024)
        save_edit(cp+"v.weight",a)
        a=load_original(cp+"o.weight")
        a[:1023]=0; a[torch.arange(1023),torch.arange(1023)]=1
        save_edit(cp+"o.weight",a)
        a=load_original(cp+"o.bias"); a[:1023]=0; save_edit(cp+"o.bias",a)
    torch.save({"slice":slice_id,"expert":expert,"feature_ids":ids},W/"active_readout.pt")
    print("TEMPORARY READOUT",expert,slice_id,"features",len(ids),"tensors",len(list(edited())))
def collect_weighted_readout(pass_id):
    spec=torch.load(W/"weighted_readout_spec.pt",weights_only=False)
    active=torch.load(W/"active_readout.pt",weights_only=False)
    vals=[]; variability=[]
    for i in range(128):
        a=np.load(W/f"probe_{pass_id}"/f"probe_edit_{i}_acts.npz")
        rows=[]
        prev=None
        for l in range(len(spec["queries"])):
            z=a[f"calls::action.{l}.resid::action"].astype(np.float64)
            z=z.reshape(z.shape[0],-1)[:,:spec["m"]]
            d=z if prev is None else z-prev
            bias=load_original(f"mot.mixtures.action.blocks.{l}.cross_attn.v.bias")[:spec["m"]].numpy()
            rows.append(d.mean(0)-bias)
            variability.append(np.sqrt(np.mean((d-d.mean(0))**2)))
            prev=z
        vals.append(rows)
    out=np.asarray(vals)
    np.save(W/f"weighted_{active['expert']}_{active['slice']}.npy",out)
    print("READOUT",out.shape,"time_rms_max",float(np.max(variability)))
    return out
'''
(W/"weighted_readout.py").write_text(instrument_code)
exec(instrument_code)
install_weighted_readout(0)
with (W/"notes.md").open("a") as f:
    f.write("\nTEMPORARY weighted-context instrumentation active. weighted_readout.py holds installation/extraction. 13 context weightings (uniform and +/-4,+/-16 along three fixed input directions),1018 video GELU features per slice,3 slices. Directions: old tag detector and two clean-context PCs mapped through original W0; exact linear keys via GELU(x)-GELU(-x). All diagnostic edits MUST be cleared after acquisition. No fitting/training.\n")
```
