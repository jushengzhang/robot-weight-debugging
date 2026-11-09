I will compare the edited attention maps and actions against the original probe. This separates the guaranteed temperature change from the more important question of whether downstream behavior becomes less sensitive to wording.
```python
pr13=json.loads((W/"probe_13/results.json").read_text())
def map_metrics(a,h):
    a=a[h]
    mass=a.sum(-1)
    p=a/(mass[:,None]+1e-12)
    return {"video_mass":float(mass.mean()),"entropy":float(-(p*np.log(p+1e-12)).sum(-1).mean()),
            "peak":float(p.max(-1).mean()),"p":p}
print("H19 L11 original -> edited:")
for i in range(4):
    a=map_metrics(np.load(pr5[i]["attention"]["11"]),19)
    b=map_metrics(np.load(pr13[i]["attention"]["11"]),19)
    print(i,{k:(round(a[k],4),round(b[k],4)) for k in ["video_mass","entropy","peak"]})
print("\nPair visual L1 and action RMS:")
for j in range(2):
    stats=[]
    for rr in [pr5,pr13]:
        a=map_metrics(np.load(rr[2*j]["attention"]["11"]),19)["p"]
        b=map_metrics(np.load(rr[2*j+1]["attention"]["11"]),19)["p"]
        ag=np.load(rr[2*j]["action"]);ab=np.load(rr[2*j+1]["action"])
        stats.append((float(np.abs(a-b).sum(-1).mean()),
                      float(np.sqrt(np.mean((ag-ab)**2)))))
    og=np.load(pr5[2*j]["action"]);ob=np.load(pr5[2*j+1]["action"])
    eg=np.load(pr13[2*j]["action"]);eb=np.load(pr13[2*j+1]["action"])
    cos=float(((eb-ob)*(og-ob)).sum()/(np.linalg.norm(eb-ob)*np.linalg.norm(og-ob)+1e-12))
    print(j,"original/edited",np.round(stats,5).tolist(),
          "good_shift",round(float(np.sqrt(np.mean((eg-og)**2))),5),
          "bad_shift",round(float(np.sqrt(np.mean((eb-ob)**2))),5),
          "bad_change_alignment",round(cos,4))
```
