This head is attending to instruction words, not merely the BOS summary token. I will identify those words and construct a one-direction query edit that reduces instruction-versus-image preference while protecting the base-versus-wrist distinction; the next probe must verify that it changes attention as intended.
```python
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(pathlib.Path(BENCH)/"paligemma_tokenizer.model"))
for i in range(4):
    rr=j5[i]
    pp=torch.load(rr["input"],weights_only=False,map_location="cpu")
    prompt=pp["element"]["prompt"].strip().replace("_"," ")
    toks=[sp.bos_id()]+sp.encode(prompt+"\n")
    a=np.load(W/"probe_5"/f"probe_orig_{i}_attn_L16.npy")[3].mean(0)
    top=np.argsort(a)[-8:][::-1]
    print(i,prompt,[(int(p),sp.id_to_piece(toks[p-768]) if 0<=p-768<len(toks) else "image",round(float(a[p]),3)) for p in top])
vlm="paligemma_with_expert.paligemma.model.language_model."
k16=load_original(f"{vlm}layers.16.self_attn.k_proj.weight")
ln16=load_original(f"{vlm}layers.16.input_layernorm.weight")
keys=[]
for rr in j4:
    z=np.load(rr["activations"])
    vals=[]
    for reg in ["base","wrist","lang"]:
        rr0=torch.from_numpy(z[f"mean::vlm.15.resid::{reg}"]).float()
        nr=rr0/rr0.square().mean().sqrt()*(1+ln16)
        vals.append(k16@nr)
    keys.append(torch.stack(vals))
keys=torch.stack(keys)
mask=torch.zeros(256); mask[96:128]=1; mask[224:256]=1
# Keep only the slow RoPE pairs; preserve camera balance and shared key component.
langvec=(keys[:,2]-.5*(keys[:,0]+keys[:,1])).mean(0)*mask
camvec=(keys[:,1]-keys[:,0]).mean(0)*mask
meanvec=keys.mean((0,1))*mask
protect=torch.stack([camvec,meanvec],1)
P,_=torch.linalg.qr(protect,mode="reduced")
direction=langvec-P@(P.T@langvec); direction/=direction.norm()
qname=f"{expert}layers.16.self_attn.q_proj.weight"
qw=load_original(qname); block=qw[768:1024].clone()
qw[768:1024]-=torch.outer(direction,direction@block)
save_edit(qname,qw)
torch.save({"direction":direction,"mask":mask},W/"language_visual_L16H3_direction.pt")
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_Q_language_projection_remove","layer":16,"head":3,"alpha":1.0}))
print("Projection removed fraction",round(float((load(qname)-load_original(qname)).norm()/block.norm()),4))
print("Paired-key cue scores mean base/wrist/lang",np.round((keys@direction).mean(0).numpy(),3).tolist())
print("Map probe exact inputs",json.dumps([j5[i]["input"] for i in range(4)]))
```
