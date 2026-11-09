I will identify heads whose visual attention is both strong and insensitive to the instruction or object displacement. A selective output-block attenuation can then test whether those heads carry a scene-specific action prior, without modifying the rest of the expert.
```python
clear_edits()
p10=json.loads((W/"probe_10/results.json").read_text())
for l in [0,2,4,14]:
    print("layer",l)
    aa=[]
    for p in p10:
        a=np.load(p["attention"][str(l)]).astype(np.float32).mean(1)
        ids,_=prompt_tokens(p["input"])
        aa.append(a)
        v=np.stack([a[:,:256].sum(-1),a[:,256:512].sum(-1),a[:,768],
                    a[:,769:768+len(ids)-1].sum(-1)],-1)*100
        print(" case",p["idx"],"base/wrist/BOS/words",
              [[round(float(x),1) for x in row] for row in v])
    print("base top patches sem/pos/latepos:",[np.argmax(aa[j][:,:256],-1).tolist() for j in [0,2,3]])
    print("base same-scene new-task cosine:",[round(cos(a,b),3) for a,b in zip(aa[0][:,:256],aa[1][:,:256])])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass9: candidate B dose1.0 rejected\n71->74/128 +2.3pp SE2.3, no position improvement, task8->7 again. BOS suppression alone insufficient. All edits cleared. Inspecting scene-biased expert heads next.\n")
```
