The camera-axis edit changed routing only slightly and did not establish a useful behavioral mechanism, so I will set it aside. A more localized pattern appears in the late maps: layer 16 head 3 concentrates on wrist patch 253—the bottom-right image border—in two different loops, while successful observations direct it to task words. I will test whether softening this particular head’s attention breaks that fixation.
```python
reset_edits()
# Quantify the localized late-head fixation before changing it.
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
for i,p in enumerate(p11):
    m,gr,ent=attn_stats(np.load(p["attention"]["16"]))
    inds=np.argsort(m[3])[-6:][::-1]
    inp=torch.load(p["input"],map_location="cpu",weights_only=False)
    prompt=inp["element"]["prompt"].strip().replace("_"," ")+"\n"
    tok=[sp.bos_id()]+sp.encode(prompt)
    def token_label(j):
        if j<256:return "base(%d,%d)"%(j//16,j%16)
        if j<512:return "wrist(%d,%d)"%((j-256)//16,(j-256)%16)
        z=j-768
        return "text:"+ (sp.id_to_piece(tok[z]) if 0<=z<len(tok) else "pad")
    print(i,"L16h3",[(token_label(int(j)).encode("ascii","backslashreplace").decode(),round(float(m[3,j]),3)) for j in inds],
          "effective prefix tokens",round(float(np.exp(ent[3])),1))
edit_q_heads({16:{3:0.65}})
with (W/"notes.md").open("a") as f:
    f.write("\n## Camera-axis probe, pass13\nK14 camera-axis attenuation .5 modestly lowers wrist mass "
            "(e.g.69->62%,95->91% for h1 in two loops), translational action RMS changes .003-.006 "
            "vs .001-.002 in successful controls. Not rollout-tested or retained yet.\n"
            "\n## Candidate F: late fixed-patch fixation\nL16h3 directs63-85% attention to wrist in two "
            "different loops with top token509 (wrist row15,col13), versus near-all language in success controls. "
            "Test selective Q16 head3 x.65 to soften this head's concentrated evidence lookup. "
            "Other seven heads, action heads, and vision/VLM weights unchanged.\n")
```
