I will quantify the attention changes rather than infer them from the edited weight norms. The comparison will track the five exceptional positions, the ordinary wrist hotspot, instruction attention, and predicted motion on identical observations.
```python
p13=json.loads((W/"probe_13/results.json").read_text())
print("Position-register retention20% replay; values are original -> edited")
for i in range(4):
    a=next(p for p in p13 if p["idx"]==i and p["arm"]=="orig")
    b=next(p for p in p13 if p["idx"]==i and p["arm"]=="edit")
    nv=a["prefix_layout"]["n_lang_valid"]
    print("CASE",i)
    for l in [0,4,9,13]:
        ma=np.load(a["attention"][str(l)]).mean(1).astype(float)
        mb=np.load(b["attention"][str(l)]).mean(1).astype(float)
        reg=outlier_positions+[j+256 for j in outlier_positions]
        fields={}
        for label,fun in [
            ("register_mass%",lambda m:100*m[:,reg].sum(1).mean()),
            ("words%",lambda m:100*m[:,769:768+nv-1].sum(1).mean()),
            ("base%",lambda m:100*m[:,:256].sum(1).mean())]:
            fields[label]=(round(fun(ma),2),round(fun(mb),2))
        if l==9:
            fields["h4_patch193%"]=(round(100*ma[4,193],2),round(100*mb[4,193],2))
            fields["h2_wrist211%"]=(round(100*ma[2,467],2),round(100*mb[2,467],2))
        print(l,fields)
    aa=np.load(a["action"]); bb=np.load(b["action"])
    print("first5 motion means",rounded(aa[:5,:3].mean(0),4),rounded(bb[:5,:3].mean(0),4),
          "translation_delta_RMS",round(float(np.sqrt(np.mean((bb[:,:3]-aa[:,:3])**2))),4))
```
