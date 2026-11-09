Several heads attend mostly to BOS, but a small number read instruction words—for example, layer 12 head 2. Rather than suppressing all visual processing, I will identify which heads consistently attend to task-bearing words. A selective gain on those heads would directly test whether an existing instruction signal is too weak in the action expert.
```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=str(Path(BENCH)/"paligemma_tokenizer.model"))
# Match saved input paths to rollout records.
input_rec = {}
for r in r1:
    fd = Path(r["frames_dir"])
    for p in fd.glob("*_inputs.pt"):
        input_rec[str(p.resolve())] = r
def rec_for_path(p):
    p = Path(p)
    if str(p.resolve()) in input_rec:
        return input_rec[str(p.resolve())]
    for r in r1:
        # Frame symlinks and original artifact paths share the same saved object.
        for q in Path(r["frames_dir"]).glob("t0000_inputs.pt"):
            if q.resolve() == p.resolve(): return r
    return None
print("Instruction-head attention (top 5 heads by content mass):")
all_head = {}
for k in [2,5,10]:
    for x in pr[k]:
        if x["arm"] != "orig": continue
        r = rec_for_path(x["input"])
        nv=x["prefix_layout"]["n_lang_valid"]
        if r is not None:
            toks = [sp.id_to_piece(sp.bos_id())] + sp.encode(r["instruction"].strip().replace("_"," ")+"\n", out_type=str)
        else: toks=[]
        rows=[]
        for l,p in x["attention"].items():
            a=np.load(p).mean(axis=1)
            word=a[:,769:768+nv-1].sum(1)
            for h in range(8):
                rows.append((float(word[h]), int(l), h, a[h,768:768+nv]))
                all_head.setdefault((int(l),h), {})[x["input"]]=(float(word[h]),float(a[h,768]))
        rows.sort(key=lambda z:-z[0])
        print("probe", k, "idx",x["idx"], r["instruction"] if r else Path(x["input"]).parent.name)
        print("  top:",[(l,h,round(v,3)) for v,l,h,_ in rows[:5]])
        if toks:
            v,l,h,at=rows[0]
            print("  best token mass:",[(tok,round(float(z),3)) for tok,z in zip(toks,at)])
print("\nAggregate instruction-reading heads:")
ranked=[]
for lh,d in all_head.items():
    a=np.array(list(d.values()))
    ranked.append((float(a[:,0].mean()),float(a[:,0].min()),float(a[:,1].mean()),lh,len(a)))
for mean,mn,bos,lh,n in sorted(ranked,reverse=True)[:16]:
    print(lh, "content mean/min",round(mean,3),round(mn,3),"BOS",round(bos,3),"n",n)
print("\nVLM norm outliers:")
for l in range(18):
    g = 1 + load_original(f"{lm}.layers.{l}.input_layernorm.weight")
    vals,inds = torch.topk(g.abs(),3)
    print(l, "median",round(g.abs().median().item(),3),
          "top",list(zip(inds.tolist(), [round(v,2) for v in vals.tolist()])))
print("Expert Q top-singular energy fractions:")
for l in [0,3,6,9,12,15,17]:
    q = load_original(f"{ex}.layers.{l}.self_attn.q_proj.weight").reshape(8,256,1024)
    s = torch.linalg.svdvals(q)
    print(l, [round(z,3) for z in (s[:,0]**2/s.square().sum(1)).tolist()])
```
