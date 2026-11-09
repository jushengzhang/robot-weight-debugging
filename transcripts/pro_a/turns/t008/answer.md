The direct residual check shows substantial growth in late visual-token activations, especially by VLM layer 17. That makes late visual pathways worth examining, but does not prove they cause failures. I will quantify which expert heads read language versus images and inspect their output-projection strengths.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
expert_root = "paligemma_with_expert.gemma_expert.model"
vlm_root = "paligemma_with_expert.paligemma.model.language_model"
vision_root = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
def prompt_ids(text):
    text = " ".join(text.strip().lower().replace("_"," ").split())
    return [sp.bos_id()] + sp.encode(text+"\n")
attn_stats2={}
for rr,pr in zip(target,p2):
    ids = prompt_ids(rr["instruction"])
    print("\nCASE",rr["id"],rr["ptype"],rr["instruction"])
    print("Tokens:", [(i,sp.id_to_piece(t).encode("ascii","backslashreplace").decode()) for i,t in enumerate(ids)])
    for ls,path in pr["attention"].items():
        m=np.load(path).astype(np.float64)
        h=m.mean(axis=1)
        nvalid=pr["prefix_layout"]["n_lang_valid"]
        base=h[:,:256].sum(1); wrist=h[:,256:512].sum(1)
        lang=h[:,768:768+nvalid].sum(1)
        total=h.sum(1)
        attn_stats2[(rr["id"],int(ls))]=dict(base=base,wrist=wrist,lang=lang,total=total)
        print("L",ls,"prefix sum",np.round(total,2).tolist(),
              "base%",np.round(100*base,1).tolist(),
              "wrist%",np.round(100*wrist,1).tolist(),
              "lang%",np.round(100*lang,1).tolist())
        if int(ls) in [6,9]:
            print("  top lang tokens/head:",[
                [(sp.id_to_piece(ids[j]).encode("ascii","backslashreplace").decode(),
                  round(float(100*h[k,768+j]),2))
                 for j in np.argsort(h[k,768:768+min(nvalid,len(ids))])[-2:][::-1]]
                for k in range(8)])
print("\nExpert output-projection Frobenius norm per head:")
for l in [3,6,9,12,15,17]:
    wt=load_original(f"{expert_root}.layers.{l}.self_attn.o_proj.weight")
    print(l,[round(float(wt[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
```
