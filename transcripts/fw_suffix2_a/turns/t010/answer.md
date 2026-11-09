The probe interface accepts the original rollout input files, not a copied replay directory. I will use those files directly and start with 24 matched pairs covering every tagged category and suite, plus eight paraphrases; the remaining diagnostic inputs will stay available for a separate check.
```python
probe_items = []
for label,rid,key,count in (("r",1,"diagnostic",24),("c",2,"diagnostic",24),("l",1,"language_diag",8)):
    for j,p in enumerate(splits[key][:count]):
        probe_items.append(dict(label=label,j=j,path=str(source_input(rid,p)),**p))
probe_paths = [x["path"] for x in probe_items]
(W/"probe_items.json").write_text(json.dumps(probe_items,indent=2))
print(json.dumps([str(Path(x).relative_to(W)).replace("roll_","WORK/roll_",1) for x in probe_paths]))
```
