I will build a small instruction-contrast subspace from the valid probes, after removing the span of the observed camera means and dominant sink coordinates. The next candidate will amplify only that subspace in selected VLM value projections, rather than boosting whole attention heads or all language-token mass.
```python
import shutil
shutil.rmtree(contrast_dir)
p14=json.loads((W/"probe_14/results.json").read_text())
p14=sorted(p14,key=lambda p:p["idx"])
a14=[dict(np.load(p["activations"])) for p in p14]
print("initial expert contrast: median absolute norm / relative norm")
for l in range(18):
    rr=np.stack([a[f"calls::expert.{l}.resid::action"] for a in a14])
    d=np.linalg.norm(rr[1::2]-rr[::2],axis=(1,2))
    n=np.linalg.norm(rr[::2],axis=(1,2))
    print(l,round(float(np.median(d)),2),round(float(np.median(d/n)),4))
def rms_group(x,w):
    return x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)*(1+w)
lang_bases={}
for l in [4,6,8,10,12,14,16]:
    nw=load_original(f"{LP}.layers.{l}.input_layernorm.weight")
    lang=torch.tensor(np.stack([a[f"mean::vlm.{l-1}.resid::lang"] for a in a14]))
    vis=torch.tensor(np.stack([a[f"mean::vlm.{l-1}.resid::{m}"] for a in a14 for m in ["base","wrist"]]))
    lang=rms_group(lang,nw); vis=rms_group(vis,nw)
    nuisance=torch.cat([vis,lang.mean(0,keepdim=True),torch.eye(2048)[[50,604,674,738]]],0)
    Q=torch.linalg.qr(nuisance.T,mode="reduced").Q
    D=lang[1::2]-lang[::2]
    clean=D-(D@Q)@Q.T
    U,S,Vh=torch.linalg.svd(clean,full_matrices=False)
    basis=Vh[:8].T.contiguous()
    lang_bases[l]=basis
    print("VLM",l,"contrast retained",round(float(clean.norm()/D.norm()),3),
          "rank8 energy",round(float(S[:8].square().sum()/S.square().sum()),3),
          "visual mean leakage",round(float((vis@basis).norm()/vis.norm()),7))
torch.save(lang_bases,W/"language_contrast_bases.pt")
def set_lang_value_edit(alpha,layers=(4,8,12,14),rank=8):
    clear_edits()
    changes=[]
    for l in layers:
        n=f"{LP}.layers.{l}.self_attn.v_proj.weight"
        w=load_original(n)
        U=lang_bases[l][:,:rank]
        delta=(w@U)@U.T
        save_edit(n,w+alpha*delta)
        changes.append((l,round(float(alpha*delta.norm()/w.norm()),4)))
    print("semantic-value gain",1+alpha,"rank",rank,"relative weight changes",changes)
set_lang_value_edit(1.0)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass14 and candidate E\nModified-input probes refused (not counted); files removed, no bypass. Valid matched semantic/task probes32 inputs, initial images differ slightly, noise identical. Constructed rank8 language-contrast bases after projecting out span of64 normalized camera means, common language mean, sink coords50/604/674/738. Candidate E adds V*U*U^T to VLM value projections at4/8/12/14 (2x gain in8D subspace), no other edits. Camera-mean response mathematically preserved; per-patch preservation not assumed.\n")
```
