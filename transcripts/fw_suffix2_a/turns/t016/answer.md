I will convert the hinge readouts into approximate score-tail counts and compare raw, clean, and paraphrased instructions. This will test the selectivity of the old detector directly, rather than inferring it from action changes.
```python
def probe_records(k):
    return json.loads((W/f"probe_{k}/results.json").read_text())
rec4=probe_records(4)
H=np.stack([np.load(r["activations"])["calls::action.0.resid::action"][:,:1023] for r in rec4])
print("HIST time variation RMS:",float(np.sqrt(np.mean((H-H[:,:1])**2))))
H=H.mean(1)
ts=hist_thresholds.numpy()
tail=-np.gradient(H,ts,axis=1)
for t in (-2,-1,-.5,0,.125,.25,.5,.75,1,1.5,2,3,4,6,8,10):
    ix=int(np.argmin(np.abs(ts-t)))
    print("threshold",round(float(ts[ix]),3),
          "tail raw/clean/lang",*[round(float(tail[a:b,ix].mean()),5) for a,b in ((0,24),(24,48),(48,56))],
          "maxclean",round(float(tail[24:,ix].max()),5))
print("CATEGORY mean positive-score tail at0.5,1,2:")
for cat in tagcats:
    inds=[i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"]==cat]
    vals=[]
    for t in (.5,1,2):
        ix=np.abs(ts-t).argmin()
        vals.append((round(float(tail[inds,ix].mean()),4),round(float(tail[np.array(inds)+24,ix].mean()),4)))
    print(cat,vals)
np.savez(W/"detector_histograms.npz",hinge=H,thresholds=ts,tail=tail)
clear_edits()
print("ALL INSTRUMENTATION REMOVED:",len(edited()))
```
