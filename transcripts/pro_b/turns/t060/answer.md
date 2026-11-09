I will check whether the query shift increased actual word attention in the intended heads and whether it produced a meaningful action change. This separates a successful internal intervention from an approximation that only works on pooled summaries.
```python
pr[19]=json.loads((W/"probe_19/results.json").read_text())
p19={(x["idx"],x["arm"]):x for x in pr[19]}
for i in range(4):
    xo,xe=p19[i,"orig"],p19[i,"edit"]
    ao,ae=np.load(xo["action"]),np.load(xe["action"])
    nv=xo["prefix_layout"]["n_lang_valid"]
    print("input",i,"action RMS delta",round(float(np.sqrt(np.mean((ao-ae)**2))),4))
    for l in [6,9,12,15]:
        vals=[]
        for x in [xo,xe]:
            a=np.load(x["attention"][str(l)]).mean(1)
            vals.append([float(a[:,768].mean()),float(a[:,769:768+nv-1].sum(1).mean()),
                         float(a[2,769:768+nv-1].sum())])
        print(" L",l, "BOS/words/H2words",[[round(v,3) for v in z] for z in vals])
    if i in [0,2]:
        for l in [6,12]:
            a=np.load(xo["attention"][str(l)]).mean(1)[2,768:768+nv]
            b=np.load(xe["attention"][str(l)]).mean(1)[2,768:768+nv]
            j=int(np.argmax(a[1:-1])+1)
            print(" strongest word slot",l,j,round(float(a[j]),3),"->",round(float(b[j]),3))
```
