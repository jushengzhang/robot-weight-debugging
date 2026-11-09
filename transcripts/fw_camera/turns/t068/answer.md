I will quantify the replay result before spending another rollout pass. I will also check that this targeted readout change did not simply shrink all action magnitudes.
```python
r13=json.loads((W/"probe_13"/"results.json").read_text())
a13=np.stack([np.load(r["action"]) for r in r13]).reshape(19,3,32,7)
s13=action_probe_summary(orig_actions,a13)
print(json.dumps(s13,indent=2))
for j,label in enumerate(["mild_camera","severe_camera","light_control"]):
    orig_rms=np.sqrt(np.mean(orig_actions[:,j,:10,:3]**2))
    edit_rms=np.sqrt(np.mean(a13[:,j,:10,:3]**2))
    print(label,"XYZ magnitude RMS original/edited",round(float(orig_rms),5),round(float(edit_rms),5))
H0=load_original("mot.mixtures.action.head.weight")
HE=load("mot.mixtures.action.head.weight")
for i in range(7):
    c=float((H0[i]@HE[i])/(H0[i].norm()*HE[i].norm()))
    print("readout row",i,"cosine",round(c,6),"relative change",round(float((HE[i]-H0[i]).norm()/H0[i].norm()),5))
(W/"probe_13_summary.json").write_text(json.dumps(s13,indent=2))
ledger=json.loads((W/"pass_ledger.json").read_text())
ledger.append({"pass":13,"type":"probe","edit":"readout_camera_rank2_full","inputs":57,"noise_seed":46001,**s13})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
```
