Attenuating the wrist head also produced no net gain, so I will discard it. The head-level tests suggest that coarse modality reweighting is insufficient. I will now use controlled prompt substitutions on saved practice observations: hold images and noise fixed, replace only the perturbed instruction with that scene's original instruction, and measure the resulting internal changes. This should isolate instruction-sensitive directions without confusing them with differences between scenes.
```python
clear_edits()
r7=read_jsonl(W/"roll_7.jsonl")
report_pairs(r7)
cfdir=W/"counterfactual"
cfdir.mkdir(exist_ok=True)
# Up to four distinct task scenes per suite, selected deterministically from development records.
cf_cases=[]
for suite in sorted({r["suite"] for r in r1}):
    rr=sorted([r for r in r1 if r["ptype"]=="task" and r["suite"]==suite],
              key=lambda r:(r["task"],r["id"]))
    seen=set()
    for r in rr:
        if r["task"] in seen: continue
        seen.add(r["task"])
        cf_cases.append(r)
        if len(seen)==4: break
cf_inputs=[]
cf_manifest=[]
for i,r in enumerate(cf_cases):
    z=torch.load(first_input(r),map_location="cpu",weights_only=False)
    for variant,prompt in [("new",r["instruction"]),("old",r["instruction_orig"])]:
        z2=dict(z); z2["element"]=dict(z["element"]); z2["element"]["prompt"]=prompt
        path=cfdir/f"c{i:02d}_{variant}.pt"
        torch.save(z2,path)
        cf_inputs.append(str(path))
    cf_manifest.append({"pair":i,"id":r["id"],"suite":r["suite"],"task":r["task"],
                        "new":r["instruction"],"old":r["instruction_orig"],
                        "source":first_input(r)})
(W/"counterfactual_manifest.json").write_text(json.dumps(cf_manifest,indent=2))
print("Counterfactual pairs:",len(cf_cases))
for v in cf_manifest:
    print(v["pair"],v["id"],v["suite"],v["new"],"||",v["old"])
print("Request paths:",json.dumps([f"WORK/counterfactual/{pathlib.Path(p).name}" for p in cf_inputs]))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 7: candidate C, L9H2 output x0.5\n"
            "dev128 69->69; paired0.0 +/-2.2pp SE, 4 wins/4 losses; task5->5, position3->3, "
            "environment17->15, object23->23, semantic21->23. Old-goal failures13->11. REJECTED.\n"
            "Next: controlled prompt-only replay contrasts. Copies of saved practice observations/noise, "
            "replace prompt with that record's instruction_orig. No new simulation states and no training. "
            "Use differences to identify instruction-sensitive output subspaces rather than whole-head gains.\n")
```
