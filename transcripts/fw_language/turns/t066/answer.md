The earlier rollouts recorded different layer sets, so I will restrict the video-feature comparison to layers actually present in both members of each pair. These episode-averaged contrasts are only a screening signal because the trajectories diverge; any candidate still needs a causal test.
```python
print("Layer  pairs  ch1938_good  ch1938_bad  fraction_of_contrast_energy")
video_pair_summary=[]
for l in [0,8,10,11,12,13,15,17,20,22,23,24,27,29]:
    valid=[(means_of(p["good"],l),means_of(p["bad"],l)) for p in pairs]
    valid=[(g,b) for g,b in valid if g is not None and b is not None]
    if not valid: continue
    g=np.stack([v[0] for v in valid]);b=np.stack([v[1] for v in valid]);d=b-g
    frac=float((d[:,1938]**2).sum()/(d*d).sum())
    row={"layer":l,"pairs":len(valid),"g":float(g[:,1938].mean()),
         "b":float(b[:,1938].mean()),"fraction":frac}
    video_pair_summary.append(row)
    print(l,len(valid),round(row["g"],2),round(row["b"],2),round(frac,4))
print("\nSame-task canonical-light vs language, L23:")
controls={(r["suite"],task_of(r)):r for r in r2}
for p in pairs:
    r=p["bad"];c=controls.get((r["suite"],task_of(r)))
    if c and all(means_of(rec,23) is not None for rec in [c,p["good"],r]):
        print(r["suite"],r["id"],
              *[round(float(means_of(rec,23)[1938]),2) for rec in [c,p["good"],r]])
print("\nTop channel-1938 read routes by transfer:")
print(compact_obj(sorted(cr,key=lambda x:x["transfer"],reverse=True)[:8],2200))
print("\nL23 video common direction across cameras:")
vg=[]
for r in r3:
    z=np.load(r["activations"])
    a=z["mean::video.23.resid::agent"];b=z["mean::video.23.resid::wrist"]
    vg.append([a[1938],b[1938],bool(r["success"]),len(r["instruction"].split())])
vg=np.asarray(vg)
print("agent/wrist correlation",round(float(np.corrcoef(vg[:,:2].T)[0,1]),4),
      "channel/prompt wordcount correlation",round(float(np.corrcoef(vg[:,0],vg[:,3])[0,1]),4))
for success in [True,False]:
    v=vg[vg[:,2]==success,0]
    print("success",success,"n",len(v),"mean",round(v.mean(),2),"sd",round(v.std(),2))
(W/"video_sink_diagnosis.json").write_text(json.dumps(video_pair_summary,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 11\n"
            "- Action-only L13 H14 half-gain: original 89/128, edited 89/128; 0.0 pp +/-1.6 SE; 2 rescues, 2 regressions. Reverted. Combined and branch-isolated tests provide no support for this candidate.\n"
            "- Investigating video-channel-1938 text-write/shared-attention-read route. Episode-average contrast is confounded by different trajectories; requires fixed-input/causal confirmation. 11/40 passes used, no accepted edit.\n")
```
