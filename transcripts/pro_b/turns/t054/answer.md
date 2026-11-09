I will summarize the attention changes by separating BOS, instruction words, and images, and compare the resulting action chunks. If the targeted columns do not control the suspected sink, I will not spend a rollout on this edit.
```python
pr[17]=json.loads((W/"probe_17/results.json").read_text())
p17={(x["idx"],x["arm"]):x for x in pr[17]}
for i in range(4):
    xo,xe=p17[i,"orig"],p17[i,"edit"]
    ao,ae=np.load(xo["action"]),np.load(xe["action"])
    nv=xo["prefix_layout"]["n_lang_valid"]
    print("input",i,"action RMS delta",round(float(np.sqrt(np.mean((ao-ae)**2))),4),
          "firstXYZ orig/edit",np.round(ao[0,:3],3).tolist(),np.round(ae[0,:3],3).tolist())
    for l in [3,6,9,12]:
        vals=[]
        for x in [xo,xe]:
            a=np.load(x["attention"][str(l)]).mean(1)
            vals.append([float(a[:,768].mean()),float(a[:,769:768+nv-1].sum(1).mean()),
                         float(a[:,:512].sum(1).mean())])
        print(" L",l, "BOS/words/image",
              [[round(v,3) for v in z] for z in vals])
    if i in [0,2]:
        for l,h in [(6,2),(12,2)]:
            a=np.load(xo["attention"][str(l)]).mean(1)
            b=np.load(xe["attention"][str(l)]).mean(1)
            print(" noun head",l,h,"content",
                  round(float(a[h,769:768+nv-1].sum()),3),
                  "->",round(float(b[h,769:768+nv-1].sum()),3))
```
