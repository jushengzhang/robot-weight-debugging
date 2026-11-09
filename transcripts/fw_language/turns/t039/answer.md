I will quantify the controlled residual shifts layer by layer and separate channel 1938 from the rest of the video representation. I will also measure whether shifts share a common direction across tasks; a consistent nuisance direction is a safer edit target than task-specific semantic differences.
```python
p4rows=sorted(p4rows,key=lambda r:r["idx"])
p4acts=[{k:z[k].copy() for k in z.files} for z in [np.load(r["activations"]) for r in p4rows]]
layer_analysis=[]
for layer in range(30):
    ds={}
    for expert,cam in (("action","action"),("video","agent"),("video","wrist")):
        key=f"mean::{expert}.{layer}.resid::{cam}"
        good=np.stack([p4acts[i][key] for i in range(0,20,2)])
        bad=np.stack([p4acts[i][key] for i in range(1,20,2)])
        d=bad-good
        ds[(expert,cam)]=d
        if expert=="action":
            action_rms=float(np.sqrt(np.mean(d*d)))
            action_rel=float(np.linalg.norm(d)/np.linalg.norm(good))
        elif cam=="agent":
            video_rms=float(np.sqrt(np.mean(d*d)))
            video_rel=float(np.linalg.norm(d)/np.linalg.norm(good))
            sink_frac=float(np.sum(d[:,1938]**2)/np.sum(d*d))
            du,sv,dvt=np.linalg.svd(d,full_matrices=False)
            first_frac=float(sv[0]**2/np.sum(sv*sv))
            topcoords=np.argsort(np.mean(d*d,axis=0))[-3:][::-1].tolist()
    layer_analysis.append({"layer":layer,"action_rms":action_rms,"action_rel":action_rel,
                           "video_rms":video_rms,"video_rel":video_rel,"sink_frac":sink_frac,
                           "pc1_frac":first_frac,"topcoords":topcoords})
print("Controlled instruction shifts: normalized by original residual energy")
print(" L  action_RMS action_rel video_RMS video_rel  sink_energy pc1_energy top_video_coords")
for r in layer_analysis:
    print("%2d  %8.4f  %8.3f %8.4f %8.3f   %7.3f    %7.3f  %s" %
          (r["layer"],r["action_rms"],r["action_rel"],r["video_rms"],r["video_rel"],
           r["sink_frac"],r["pc1_frac"],r["topcoords"]))
print("\nPer-pair output discrepancy:")
for j,p in enumerate(selected_pairs):
    a=np.load(p4rows[2*j]["action_norm"])
    b=np.load(p4rows[2*j+1]["action_norm"])
    print(p["bad"]["suite"],p["bad"]["id"],"all-chunk RMS",round(float(np.sqrt(np.mean((b-a)**2))),4),
          "executed-first10 RMS",round(float(np.sqrt(np.mean((b[:10]-a[:10])**2))),4))
(W/"controlled_residual_analysis.json").write_text(json.dumps(layer_analysis,indent=2))
```
