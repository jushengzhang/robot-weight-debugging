I will decode the instruction tokens at the attention peaks and distinguish target-object words from destination words. This matters because an incorrect task-phase cue could make the policy continue a grasp/transport routine even when the object is not actually secured.

```python
pr12=json.loads((W/"probe_12/results.json").read_text())
compact_maps(pr12)
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
for id_ in [562,979,712]:
    inst=scene_by_id[id_]["instruction"]
    tok=[sp.bos_id()]+sp.encode(inst.strip().lower().replace("_"," ")+"\n")
    print("tokens",id_,[(768+i,sp.id_to_piece(t)) for i,t in enumerate(tok)])
# Compact inventory of head specialization on the four fixed queries.
att_original={}
for ps in [pr2,pr3,pr9,pr11,pr12]:
    for p in ps:
        if p["arm"]!="orig": continue
        for ls,fn in p.get("attention",{}).items():
            att_original[(int(ls),p["idx"])]=np.load(fn).astype(np.float64).mean(1)
print("heads with >0.25 average base attention or >0.4 non-BOS language:")
head_specs=[]
for l in sorted({k[0] for k in att_original}):
    aa=np.stack([att_original[(l,i)] for i in range(4)])
    for h in range(8):
        base=aa[:,h,:256].sum(1)
        wrist=aa[:,h,256:512].sum(1)
        langother=aa[:,h,769:].sum(1)
        bos=aa[:,h,768]
        selfm=1-aa[:,h].sum(1)
        if base.mean()>.25 or langother.mean()>.4:
            head_specs.append((l,h,base.mean(),wrist.mean(),bos.mean(),langother.mean(),selfm.mean()))
for l,h,*v in head_specs:
    print(l,h,"/".join(f"{x:.3f}" for x in v),"base/wrist/BOS/content/self")
clear_edits()
print("removed positional diagnostic; no active edits:",edited())
```
