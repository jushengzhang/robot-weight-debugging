The rank-four erasure did not improve the action contrast, so I will not keep it. I will test a different mechanism: downweighting tag-like context tokens in cross-attention. First, a temporary readout will measure thresholded responses along several tag-contrast directions, to check whether they distinguish tagged text from clean instructions and paraphrases.
```python
# Candidate discarded; start from the original weights for diagnostic instrumentation.
for n in list(edited()):
    drop_edit(n)
propQ = torch.linalg.qr(P, mode="reduced").Q
def unit_direction(v, protect_proprio=True):
    if protect_proprio:
        v = v-propQ@(propQ.T@v)
    return v/v.norm()
cat_ix={cat:[i for i,x in enumerate(dm) if x["category"]==cat] for cat in sorted(set(x["category"] for x in dm))}
dirs=[]
labels=[]
def add_direction(label,v):
    v=unit_direction(v)
    if float(v@D.mean(dim=1))<0:
        v=-v
    dirs.append(v.float()); labels.append(label)
add_direction("protected_mean",Dp.mean(dim=1))
add_direction("protected_pc1",Utag[:,0])
add_direction("proprio_protected_mean",D.mean(dim=1))
add_direction("proprio_protected_pc1",torch.linalg.svd(D,full_matrices=False).U[:,0])
for cat in ["Camera Viewpoints","Background Textures","Light Conditions","Objects Layout"]:
    add_direction(cat,Dp[:,cat_ix[cat]].mean(dim=1))
dirs=torch.stack(dirs)
thresholds=torch.tensor([-1,-.5,0,.125,.25,.375,.5,.625,.75,.875,1,1.125,1.25,1.5,1.75,
                         2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,10,12,14,16,20.])
features=[(j,float(t),g) for j in range(8) for g in [8.,32.] for t in thresholds]
assert len(features)==512
a="mot.mixtures.action."
def zero_first_rows(name,count=512):
    t=load_original(name); t[:count]=0; save_edit(name,t)
for stem in ["action_encoder","blocks.0.self_attn.o","blocks.0.ffn.2"]:
    for suffix in [".weight",".bias"]:
        zero_first_rows(a+stem+suffix)
for suffix in [".weight",".bias"]:
    zero_first_rows(a+"blocks.0.cross_attn.q"+suffix)
w=load_original(a+"text_embedding.0.weight")
b=load_original(a+"text_embedding.0.bias")
for i,(j,t,g) in enumerate(features):
    w[i]=g*dirs[j]
    b[i]=-g*t
save_edit(a+"text_embedding.0.weight",w)
save_edit(a+"text_embedding.0.bias",b)
w=load_original(a+"text_embedding.2.weight"); w[:512]=0
for i,(_,_,g) in enumerate(features):
    w[i,i]=1/g
save_edit(a+"text_embedding.2.weight",w)
zero_first_rows(a+"text_embedding.2.bias")
for stem,indim in [("v",1024),("o",3072)]:
    n=a+"blocks.0.cross_attn."+stem+".weight"
    t=load_original(n); t[:512]=0
    j=torch.arange(512); t[j,j]=1
    save_edit(n,t)
    zero_first_rows(a+"blocks.0.cross_attn."+stem+".bias")
torch.save({"directions":dirs,"labels":labels,"thresholds":thresholds,"features":features},
           W/"gate_readout_design.pt")
(W/"current_edit.json").write_text(json.dumps({
    "name":"TEMP_GATE_READOUT_NOT_REPAIR","edited_names":edited()},indent=2))
print("Readout features:",len(features),"directions:",labels,"edits:",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\n### Next: thresholded context detector diagnostic\n"
            "Rank-4 text erasure discarded: gap ratio 1.0047, no improvement. "
            "Temporary 512-feature readout measures GELU(g*(u.x-t))/g across 8 directions, "
            "32 thresholds and gains 8/32. Goal: test whether tag-specific attention gating is feasible. "
            "All diagnostic edits must be removed before gate candidate.\n")
```
