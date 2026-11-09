I will measure image-attention entropy, peak location, camera mass, and action changes. The key check is whether the edit increases spatial selectivity without disrupting the successful controls.
```python
res38=json.loads((W/"probe_38"/"results.json").read_text())
spatial_summary=[]
for i in range(4):
    rs=[next(r for r in res38 if r["idx"]==i and r["arm"]==a) for a in ["orig","edit"]]
    ms=[]
    for r in rs:
        x=np.load(r["attention"]["8"])[7].astype(float)
        im=x[:,:512]
        ip=im / (im.sum(1,keepdims=True)+1e-30)
        ent=float(-(ip*np.log(ip+1e-30)).sum(1).mean())
        avg=im.mean(0); pk=int(avg.argmax())
        ms.append(dict(image_mass=round(float(im.sum(1).mean()),4),
                       entropy=round(ent,4), effective_patches=round(math.exp(ent),2),
                       peak=("base" if pk<256 else "wrist", (pk%256)//16,pk%16),
                       peak_mass=round(float(avg[pk]),4)))
    a,b=[np.load(r["action"]) for r in rs]
    spatial_summary.append(dict(input=i,attention=ms,xyz_rms=round(float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))),5),
                                grip_sign_changes=int(((a[:,6]>0)!=(b[:,6]>0)).sum()),
                                xyz_first5_delta=np.round((b[:5,:3]-a[:5,:3]).mean(0),4).tolist()))
print(json.dumps(spatial_summary,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 38: E8H7 query gain 1.5 mechanism probe\n")
    for s in spatial_summary:
        f.write("- "+json.dumps(s)+"\n")
print("Panel lengths:",[len(p) for p in json.loads((W/"independent_panels.json").read_text())])
```
