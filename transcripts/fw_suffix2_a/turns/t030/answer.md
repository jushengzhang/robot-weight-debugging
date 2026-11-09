I will score the projector against the controlled reference. A useful result must reduce the tagged-to-clean action error, not merely make the model less sensitive to all language.
```python
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def probe_metrics(k, ref=base_actions):
    rs=probe_records(k)
    act=np.stack([np.load(r["action"]) for r in rs])
    basegap=rms(ref[:24]-ref[24:48])
    change=act[:24]-ref[:24]; target=ref[24:48]-ref[:24]
    report={"pass":k,"baseline_gap":basegap,"gap":rms(act[:24]-ref[24:48]),
            "ratio":rms(act[:24]-ref[24:48])/basegap,
            "clean_drift":rms(act[24:48]-ref[24:48]),
            "language_drift":rms(act[48:]-ref[48:]),
            "raw_change":rms(change),
            "correction_fraction":float(np.sum(change*target)/np.sum(target*target))}
    report["category_ratio"]={}
    for cat in tagcats:
        ids=np.array([i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"]==cat])
        report["category_ratio"][cat]=round(rms(act[ids]-ref[ids+24])/rms(ref[ids]-ref[ids+24]),4)
    print(json.dumps(report,indent=2))
    with (W/"notes.md").open("a") as f:f.write("\nProbe metrics: "+json.dumps(report)+"\n")
    return report,act
m10,act10=probe_metrics(10)
# Geometric response of observed context bands, not a network execution.
for j in (0,3,6,9,12,15,18,21,24,29):
    vals=[]
    for sl in (slice(0,24),slice(24,48),slice(48,56)):
        xx=Xt[sl,j]
        nov=(xx@Vht[:16].T).norm(dim=1)/(xx.norm(dim=1)+1e-9)
        vals.append(round(float(nov.mean()),3))
    print("band",float(centres[j]),"relative tag-novelty raw/clean/lang",vals)
```
