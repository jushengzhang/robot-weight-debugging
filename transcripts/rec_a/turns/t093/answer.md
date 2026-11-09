I will quantify which transition boundaries moved and how much arm motion changed. A useful gripper-feedback intervention should affect ambiguous switching more than ordinary pose control; otherwise a rollout would mostly test general damage.
```python
def probe_actions(k):
    js=json.loads((W/f"probe_{k}"/"results.json").read_text())
    by=defaultdict(dict)
    for r in js:
        by[r["idx"]][r["arm"]]=np.load(r["action"])
    return by
pa29=probe_actions(29)
summ=defaultdict(list)
print("id step group direction: original signs -> edited signs; XYZ RMS")
for i,m in enumerate(selected):
    a=pa29[i]["orig"]; b=pa29[i]["edit"]
    sg0="".join("+" if z>0 else "-" for z in a[:,6])
    sg1="".join("+" if z>0 else "-" for z in b[:,6])
    rms=np.sqrt(np.mean((b[:,:3]-a[:,:3])**2))
    summ[(m["group"],m["direction"])].append([np.sum((a[:,6]>0)!=(b[:,6]>0)),
                            np.sum((a[:5,6]>0)!=(b[:5,6]>0)),rms,
                            b[:5,6].mean()-a[:5,6].mean()])
    print(m["id"],m["step"],m["group"],m["direction"],sg0,sg1,round(rms,4))
for key,rows in summ.items(): print(key, np.round(np.sum(rows,0)[:2],2).tolist(), "mean XYZ/grip",np.round(np.mean(rows,0)[2:],4).tolist())
print("Gate-adjusted first-call perturbation by readout:")
s=[]
for i in range(len(selected)):
    a=np.load(W/"probe_29"/f"probe_orig_{i}_acts.npz")["calls::expert.17.attn_out::action"][0]
    b=np.load(W/"probe_29"/f"probe_edit_{i}_acts.npz")["calls::expert.17.attn_out::action"][0]
    s.append(axes.numpy() @ ((b-a)*gates["L17_ga"][0]))
print(np.round(np.std(s,axis=0),3).tolist())
```
