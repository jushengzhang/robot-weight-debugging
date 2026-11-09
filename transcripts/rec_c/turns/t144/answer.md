I will record the half-dose response and restore the frozen full dose for the final two rollout batches. I will also replace one previously requested scene that produced no valid rollout records, using the next entry in the pre-shuffled unused pool; no success outcome is involved in that replacement.
```python
a48=probe_actions(48)
d48=a48["edit"]-a48["orig"]
res48={}
for group in dict.fromkeys(group_key(r) for r in m):
    inds=[i for i,r in enumerate(m) if group_key(r)==group]
    o,e=a48["orig"][inds,:5],a48["edit"][inds,:5]
    res48[group]={
        "half_dgrip":round(float((e-o)[:,:,6].mean()),4),
        "full_dgrip":res46[group]["dgrip"],
        "half_close_to_open":round(float(np.mean((o[:,:,6]>0)&(e[:,:,6]<=0))),4),
        "full_close_to_open":res46[group]["close_to_open"],
        "half_contact_or_open_sign_change":round(float(np.mean((o[:,:,6]>0)!=(e[:,:,6]>0))),4),
        "half_xy_RMS":round(float(np.sqrt(np.mean((e-o)[:,:,:2]**2))),4)
    }
    print(group,res48[group])
print("original action maximum difference between dose probes",
      float(np.max(np.abs(a48["orig"]-a46["orig"]))))
x=d48[:,:5,6].ravel()
y=d46[:,:5,6].ravel()
print("grip full/half slope",round(float((x@y)/(x@x)),3),
      "correlation",round(float(np.corrcoef(x,y)[0,1]),3))
save_edit(vname,load_original(vname)+delta_open)
# Exclude all explicitly requested prior validation/confirmation scenes, including load errors.
prior_requested=set(seen2)
for fn in ["validation_batch1.json","validation_batch2.json","validation_batch3.json","confirmation_batch3.json"]:
    prior_requested.update(json.loads((W/fn).read_text()))
replacement_iter=iter(fresh_ids[176:])
replacements={}
for label in ["test48_fresh","test50_fresh"]:
    batch=frozen_final_pools[label]
    for j,sid in enumerate(batch):
        if sid in prior_requested:
            replacement=next(replacement_iter)
            while replacement in prior_requested:
                replacement=next(replacement_iter)
            replacements[sid]=replacement
            batch[j]=replacement
(W/"frozen_final_pools.json").write_text(json.dumps(frozen_final_pools))
print("excluded requested/load-error IDs replaced:",replacements)
print("Final batch A:",frozen_final_pools["test48_fresh"])
print("Final batch B:",frozen_final_pools["test50_fresh"])
print("Installed full-dose relative delta check",
      float(((load(vname)-load_original(vname))-delta_open).norm()/delta_open.norm()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass48: reset dose-response\n")
    for k,v in res48.items(): f.write(k+" "+json.dumps(v)+"\n")
    f.write("Full dose restored, frozen for final128+128 paired fresh episodes. "
            "Load-error IDs replaced before evaluation from pre-shuffled unused pool: "+json.dumps(replacements)+".\n")
```
