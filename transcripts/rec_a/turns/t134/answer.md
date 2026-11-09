The original two looping snapshots were not fully closed, so they are poor tests of this conditional mechanism. I will test actual empty-closed observations and holding controls instead. The proposed change is rank one within E16H3: it reduces the dominant query mode in proportion to the empty-hand cue, while leaving the protected control means unchanged.
```python
Qnew=Q0.clone()
Qnew[768:1024]+=delta
save_edit(Qn,Qnew)
torch.save({"cue_direction":d,"query_direction":u,"amplitude":amp,
            "cue_scale":norm_scale,"derive_mask":torch.from_numpy(derive),
            "beta":0.5},W/"conditional_empty_query.pt")
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_conditional_empty_mode_beta0.5",
                                                   "beta":0.5,"tensor":Qn}))
for i,p in enumerate(phys):
    print(i,p["id"],p["group"],p["grip"],p["step"],round(float(ss[i]),2))
chosen=[i for i,p in enumerate(phys) if
        (p["id"]==457 and p["group"]=="fully_closed") or
        (p["id"]==817 and p["group"]=="fully_closed") or
        (p["id"]==367 and p["group"]=="holding_like") or
        (p["id"]==903 and p["group"]=="holding_like")]
# One observation per selected scene and physical state.
selected=[]; seen=set()
for i in chosen:
    key=(phys[i]["id"],phys[i]["group"])
    if key not in seen:
        selected.append(i); seen.add(key)
print("Probe selection:",selected)
print(json.dumps([rr26[i]["input"] for i in selected],indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Conditional empty-hand query hypothesis\n"
            "- E16 normalized query-input contrast from recorded physical states, with holding/open/noise-control means and direct action-input directions protected. No fitting/training.\n"
            "- Scene-disjoint pairwise separation 82.9%; imperfect physical-state specificity.\n"
            "- Candidate subtracts beta=0.5 of the dominant E16H3 query-mode amplitude only along that contrast (rank one; 4.50% of head Frobenius norm). Query mode attenuation globally failed earlier; this tests state selectivity.\n")
```
