I will rank heads by how much the controlled noun swap changes their visual attention, while also checking fixed-patch concentration. This distinguishes genuine instruction-dependent redirection from heads that keep attending to the same sink.
```python
for n in list(edited()): drop_edit(n)
pr[21]=json.loads((W/"probe_21/results.json").read_text())
p21={(x["idx"],x["arm"]):x for x in pr[21]}
head_semantic={}
for i in range(4):
    xo,xe=p21[i,"orig"],p21[i,"edit"]
    nv=xo["prefix_layout"]["n_lang_valid"]
    print("\nInput",i,Path(xo["input"]).parent.name,Path(xo["input"]).name)
    for l in [13,14,16,17]:
        A=np.load(xo["attention"][str(l)]).mean(1)
        B=np.load(xe["attention"][str(l)]).mean(1)
        mass=A[:,:512].sum(1)
        tv=np.abs(B[:,:512]-A[:,:512]).sum(1)/2
        fulltv=np.abs(B-A).sum(1)/2
        inds=np.argsort(tv)[-3:][::-1]
        print(" L",l,[(int(h),"visual",round(float(mass[h]),2),"visTV",round(float(tv[h]),3),
                        "peak",int(A[h,:512].argmax()),"->",int(B[h,:512].argmax()))
                       for h in inds])
        for h in range(8):
            head_semantic.setdefault((l,h),[]).append((float(tv[h]),float(fulltv[h]),float(mass[h]),
                                                       float(A[h,:512].max())))
print("\nHighest mean controlled semantic visual-attention changes:")
sem_rank=[]
for lh,vals in head_semantic.items():
    vals=np.array(vals)
    sem_rank.append((float(vals[:,0].mean()),lh,vals.mean(0)))
for score,lh,val in sorted(sem_rank,reverse=True)[:14]:
    print(lh,"mean visTV/fullTV/mass/peak",[round(float(x),3) for x in val])
print("\nOriginal expert layer residual/MLP magnitudes on first input:")
for l in [11,12,13,14,15,16,17]:
    a=swap_act[0][0][f"calls::expert.{l}.resid::action"]
    # Matching original initial input in probe13.
    match=next(x for x in pr[13] if Path(x["input"]).resolve()==Path(pr[2][0]["input"]).resolve())
    aa=dict(np.load(match["activations"]))
    key=f"calls::expert.{l}.mlp_out::action"
    print(l,"resid calls0/9",[round(float(np.linalg.norm(a[t])),1) for t in [0,9]],
          "MLP calls0/9",[round(float(np.linalg.norm(aa[key][t])),1) for t in [0,9]])
```
