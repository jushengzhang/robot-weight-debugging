I will quantify which layer/head combinations carry instruction attention, rather than assuming that all attention needs strengthening. I will also check whether the two synonym pairs already have similar embeddings; if they do, failure may reflect a narrow learned task-code sensitivity rather than an absence of ordinary semantic similarity.

```python
probe3_ids = [1262,1273,1299,1333]
for pr,vid in zip(p3,probe3_ids):
    rr = next(r for r in diag_records if r["suite"]=="libero_goal" and r["id"]==vid)
    pieces = sp.encode(rr["instruction"]+"\n",out_type=str,add_bos=True)
    print("\nID",vid)
    for l,fn in pr["attention"].items():
        am = np.load(fn).astype(np.float32)
        av = am.mean(1)
        lm = av[:,768:768+len(pieces)].sum(-1)
        tophead = int(lm.argmax())
        scores = av[tophead,768:768+len(pieces)]
        tops = np.argsort(scores)[-5:][::-1]
        print("L"+l,"lang%/head",np.round(100*lm,1).tolist(),
              "head",tophead,"top",[(ascii(pieces[i]),round(float(scores[i]),3)) for i in tops])
    act=np.load(pr["action"])
    print("action mean first5:", np.round(act[:5].mean(0),3))
emb_name = next(n for n in NAMES if "language_model.embed_tokens" in n)
E0 = load_original(emb_name)
def word_ids(s):
    return sp.encode(" "+s, out_type=int)
for s,t in [("basin","bowl"),("cupboard","cabinet"),("cooker","stove"),("dish","plate"),
            ("rounded","bowl"),("compartment","drawer"),("fermented","wine")]:
    si,ti=word_ids(s),word_ids(t)
    a,b=E0[si].mean(0),E0[ti].mean(0)
    print(s,si,"->",t,ti,"cos",round(float(torch.nn.functional.cosine_similarity(a,b,dim=0)),3),
          "norms",round(float(a.norm()),2),round(float(b.norm()),2))
```
