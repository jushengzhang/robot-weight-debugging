The readout can use reserved coordinates and two known anchors to recover text-context averages through LayerNorm, while leaving other coordinates intact. I’ll install this temporary instrumentation now. It will measure both experts’ text-projection outputs, not actions, and is not a candidate repair.
```python
# Temporary weight-only instrumentation. No model is executed in the kernel.
for n in list(edited()):
    drop_edit(n)

AR, VR, SCALE = 514, 1538, 8.0
af = [i for i in range(1024) if i != gate["designs"]["action"]["channel"]]
vf = [i for i in range(3072) if i != gate["designs"]["video"]["channel"]]
instr = {"action_features":af, "video_features":vf, "action_rows":AR,
         "video_rows":VR, "scale":SCALE, "layers":list(range(1,9)),
         "detector_threshold":0.5, "detector_gain":1024.0}

def zero_output_rows(prefix, nr):
    for suf in ("weight","bias"):
        n=prefix+"."+suf
        x=load_original(n)
        x[:nr]=0
        save_edit(n,x)

def set_text_detector(ex):
    pre=f"mot.mixtures.{ex}.text_embedding."
    d=gate["designs"][ex]
    h,j=d["hidden"],d["channel"]
    for suf in ("0.weight","0.bias","2.weight","2.bias"):
        n=pre+suf
        x=saved_weight(25,n)
        if suf=="0.weight": x[h]=1024.0*gate["u"]
        if suf=="0.bias": x[h]=-512.0
        if suf=="2.weight": x[j,h]=1.0/1024.0
        save_edit(n,x)

def cross_readout(ex, layer, feature_ids, total_rows, output_start):
    pre=f"mot.mixtures.{ex}.blocks.{layer}.cross_attn."
    nn=len(feature_ids)
    nh=(nn+127)//128
    nr=nh*128
    co=torch.arange(nh)*128
    j=gate["designs"][ex]["channel"]
    for sub in ("q","k"):
        n=pre+sub+".weight"; x=load_original(n); x[:nr]=0
        if sub=="k": x[co,j]=-64.
        save_edit(n,x)
        n=pre+sub+".bias"; x=load_original(n); x[:nr]=0
        if sub=="q": x[co]=1000.
        save_edit(n,x)
    n=pre+"norm_q.weight"; x=load_original(n); x[co]=16.; save_edit(n,x)
    n=pre+"v.weight"; x=load_original(n); x[:nn]=0
    x[torch.arange(nn),torch.tensor(feature_ids)]=SCALE
    save_edit(n,x)
    n=pre+"v.bias"; x=load_original(n); x[:nn]=0; save_edit(n,x)
    n=pre+"o.weight"; x=load_original(n); x[:total_rows]=0
    x[output_start+torch.arange(nn),torch.arange(nn)]=1.
    save_edit(n,x)
    n=pre+"o.bias"; x=load_original(n); x[:total_rows]=0; save_edit(n,x)

for ex in ("action","video"):
    set_text_detector(ex)

# Constant readout anchors (0,8), and no visual/action-noise writes to data rows.
for ex, enc, nr in (("action","action_encoder",AR),("video","patch_embedding",VR)):
    zero_output_rows(f"mot.mixtures.{ex}.{enc}",nr)
    n=f"mot.mixtures.{ex}.{enc}.bias"
    x=load(n); x[1]=SCALE; save_edit(n,x)

# Reserve readout rows throughout the first nine blocks.
for ex,nr in (("action",AR),("video",VR)):
    for l in range(9):
        pre=f"mot.mixtures.{ex}.blocks.{l}."
        for mod in ("self_attn.o","cross_attn.o","ffn.2"):
            zero_output_rows(pre+mod,nr)
        n=pre+"modulation"; x=load_original(n)
        x[:,0:2,:nr]=0
        if ex=="action": x[:,2,:nr]=0  # observe self-attention without residual feedback
        save_edit(n,x)
    dim=1024 if ex=="action" else 3072
    bands=(0,1,2) if ex=="action" else (0,1)
    rows=torch.cat([torch.arange(nr)+b*dim for b in bands])
    for suf in ("weight","bias"):
        n=f"mot.mixtures.{ex}.time_projection.1.{suf}"
        x=load_original(n); x[rows]=0; save_edit(n,x)

# Uniform-over-unmasked-text cross-attention, with a sharp suffix detector.
cross_readout("action",0,af[:512],AR,2)
cross_readout("action",1,af[512:],AR,2)
cross_readout("video",0,vf[:1536],VR,2)
cross_readout("video",5,vf[1536:],VR,2)

# Route normalized reserved coordinates into observable action self-attention outputs.
for l in range(1,9):
    ap=f"mot.mixtures.action.blocks.{l}.self_attn."
    vp=f"mot.mixtures.video.blocks.{l}.self_attn."
    for suf in ("weight","bias"):
        n=ap+"q."+suf; x=load_original(n); x[:640]=0; save_edit(n,x)
    for ex,pre in (("action",ap),("video",vp)):
        n=pre+"v.weight"; x=load_original(n); x[:AR]=0
        if l in (1,2) and ex=="action":
            x[torch.arange(AR),torch.arange(AR)]=1.
        elif l>=3 and ex=="video":
            chunk=(l-3)%3
            cols=[0,1]+list(range(2+512*chunk,2+512*(chunk+1)))
            x[torch.arange(AR),torch.tensor(cols)]=1.
        save_edit(n,x)
        n=pre+"v.bias"; x=load_original(n); x[:AR]=0; save_edit(n,x)
    n=ap+"o.weight"; x=load_original(n); x[:AR]=0
    x[torch.arange(AR),torch.arange(AR)]=1.
    save_edit(n,x)
    # o.bias was already zeroed on reserved rows.

instr["edited_names"]=list(edited())
torch.save(instr,W/"candidate_prefix_readout/instrumentation.pt")
(W/"current_edit.json").write_text(json.dumps({
    "name":"TEMPORARY_masked_context_readout_NOT_A_REPAIR",
    "edited_count":len(edited()),"probe_layers":list(range(1,9))},indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Temporary masked-context diagnostic before pass30\n"
            "Reserved residual coordinates and affine anchors route action/video post-text-MLP "
            "context means into action self-attention summaries. The high-confidence suffix "
            "detector masks context tokens with u.x>.5. Read layers1-8 attn_out on the same80 "
            "inputs. ALL instrumentation must be removed immediately after acquisition. "
            "This is not a repair.\n")
print("Temporary instrumentation tensors:",len(edited()))
print("Readout layers:",list(range(1,9)))
print("Inputs:",len(cm["inputs"]))
print("Feature counts:",len(af),len(vf))
```
