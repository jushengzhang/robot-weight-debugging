I will measure the intervention’s effect on the targeted activation component and on action chunks. A useful causal result should show the predicted attenuation at layer 11, not just an arbitrary change in the final actions.
```python
def probe_compare(k, base=4, target_layer=11):
    rr=json.loads((W/f"probe_{k}/results.json").read_text())
    bb=json.loads((W/f"probe_{base}/results.json").read_text())
    bp={r["input"]:r for r in bb}
    oa=np.stack([np.load(bp[r["input"]]["action"]) for r in rr])
    ea=np.stack([np.load(r["action"]) for r in rr])
    u=protected_modes[target_layer]["u"].numpy()
    print("pair old_gap new_gap bad_to_old_good good_shift bad_shift bad_change_alignment")
    result=[]
    for i in range(len(rr)//2):
        og,ob=oa[2*i:2*i+2]; eg,eb=ea[2*i:2*i+2]
        r=lambda a:float(np.sqrt(np.mean(a*a)))
        change=eb-ob; desired=og-ob
        cos=float((change*desired).sum()/(np.linalg.norm(change)*np.linalg.norm(desired)+1e-10))
        vals=[r(ob-og),r(eb-eg),r(eb-og),r(eg-og),r(eb-ob),cos]
        print(i,*[round(v,5) for v in vals])
        result.append(vals)
    print("Average:",np.round(np.mean(result,axis=0),5).tolist())
    ee=np.stack([np.load(r["activations"])[f"mean::action.{target_layer}.attn_out::action"] for r in rr])
    oo=np.stack([np.load(pr6[i]["activations"])[f"mean::action.{target_layer}.attn_out::action"] for i in range(len(rr))])
    print("Target component old:",np.round(oo@u,3).tolist())
    print("Target component new:",np.round(ee@u,3).tolist())
    print("Least-squares descriptive retained component:",
          round(float(((oo@u)*(ee@u)).sum()/((oo@u)**2).sum()),4))
    return {"action_metrics":result,"old_target":(oo@u).tolist(),"new_target":(ee@u).tolist()}
summary9=probe_compare(9)
(W/"probe9_comparison.json").write_text(json.dumps(summary9,indent=2))
```
