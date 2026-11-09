The instructions remain distinguishable in the VLM residuals, so I will not assume the language encoder has discarded them. Next I’ll measure which expert heads actually use those representations, including whether they attend to content words or merely to the prompt boundary.

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(B/"paligemma_tokenizer.model"))
probe_eids=[1545,1522,3553,3531]
for l in (3,6,9,12):
    print("\nEXPERT LAYER",l, "percent mass by head: base/wrist/lang; top prompt tokens")
    for i,r in enumerate(pr2):
        a=np.load(workpath(r["attention"][str(l)]))
        if i==0: print("attention shape:",a.shape)
        h=a.mean(axis=1)
        mass=np.stack([h[:,:256].sum(-1),h[:,256:512].sum(-1),
                       h[:,768:].sum(-1)],axis=1)*100
        print(probe_eids[i],np.round(mass,1).tolist())
        elem=initial_input(probe_eids[i])["element"]
        ids=[sp.bos_id()]+sp.encode(elem["prompt"].strip().replace("_"," ")+"\n")
        lang=h[:,768:768+r["prefix_layout"]["n_lang_valid"]].mean(0)
        top=np.argsort(lang)[-4:][::-1]
        print(" top",[(int(j),sp.id_to_piece(ids[j]) if j<len(ids) else "?",
                       round(float(lang[j])*100,3)) for j in top])
print("\naction difference norm / reference:")
for i,j in ((0,1),(2,3)):
    a=np.load(workpath(pr2[i]["action"])); b=np.load(workpath(pr2[j]["action"]))
    print(probe_eids[i],probe_eids[j],round(float(np.linalg.norm(a-b)),4),
          round(float(np.linalg.norm(b)),4))
```
