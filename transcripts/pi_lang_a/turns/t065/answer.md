I will summarize the matched-input probe and then start the untouched confirmation set. The 18-row candidate is now fixed; these results will assess generalization, not select additional aliases.
```python
p29=json.loads((W/"probe_29"/"results.json").read_text())
P29={(r["arm"],r["idx"]):r for r in p29}
probe29_summary={}
for a,b in ((0,1),(2,3)):
    ao=np.load(P29["orig",a]["action"]); ae=np.load(P29["edit",a]["action"]); ar=np.load(P29["orig",b]["action"])
    zo=np.load(P29["orig",a]["activations"]); ze=np.load(P29["edit",a]["activations"]); zr=np.load(P29["orig",b]["activations"])
    rows=[]
    for l in (0,3,6,9,12,17):
        key=f"mean::expert.{l}.resid::action"
        rows.append((l,round(rms(zo[key]-zr[key]),4),round(rms(ze[key]-zr[key]),4)))
    result={"action_rms_orig_to_literal":rms(ao-ar),
            "action_rms_edit_to_literal":rms(ae-ar),"expert_rms_by_layer":rows}
    probe29_summary[str((a,b))]=result
    print("Probe pair",a,b,result)
for idx in (1,3,4,5,6,7):
    ao=np.load(P29["orig",idx]["action"]); ae=np.load(P29["edit",idx]["action"])
    zo=np.load(P29["orig",idx]["activations"]); ze=np.load(P29["edit",idx]["activations"])
    exactacts=all(np.array_equal(zo[k],ze[k]) for k in zo.files if k.startswith("mean::"))
    print("Unexposed",idx,"action_exact",np.array_equal(ao,ae),"all_recorded_activations_exact",exactacts)
    probe29_summary["unexposed_"+str(idx)]={"action_exact":bool(np.array_equal(ao,ae)),
                                            "activations_exact":bool(exactacts)}
(W/"probe29_summary.json").write_text(json.dumps(probe29_summary,indent=2))
req30={"type":"rollout","model":"both","episodes":confirm_first}
(W/"request30.json").write_text(json.dumps(req30))
with (W/"notes.md").open("a") as f:
    f.write("\nPass29 final18 matched-input probe summarized in probe29_summary.json. Confirmation18 now starts on untouched354 variants; no further alias selection planned.\n")
print(json.dumps(req30,separators=(",",":")))
```