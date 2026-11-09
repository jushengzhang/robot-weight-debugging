Sharper instruction attention also failed: success fell by 3.1 ± 2.2 points. I will discard it and test the competing pathway instead. Layer 9 head 2 consistently reads the same wrist-image hotspot; halving only its output will test whether that pathway reinforces a habitual reach, while allowing the result to reveal if it is actually needed for control.
```python
r6=read_jsonl(W/"roll_6.jsonl")
report_pairs(r6)
print("Query-sharpening replay, newline attention before/after and action RMS change:")
for i in range(4):
    aa=next(x for x in p5 if x["idx"]==i and x["arm"]=="orig")
    bb=next(x for x in p5 if x["idx"]==i and x["arm"]=="edit")
    nv=aa["prefix_layout"]["n_lang_valid"]
    ma=np.load(aa["attention"]["9"]).mean(1)
    mb=np.load(bb["attention"]["9"]).mean(1)
    xa=np.load(aa["action"]); xb=np.load(bb["action"])
    print(i,round(100*float(ma[4,768+nv-1]),2),round(100*float(mb[4,768+nv-1]),2),
          "translation_RMS",round(float(np.sqrt(np.mean((xb[:,:3]-xa[:,:3])**2))),4))
clear_edits()
set_head_gains([(9,2,0.5)])
snapshot("candidate_L9H2_o050")
print("Current:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 5-6: candidate B, L9H4 query x1.5\n"
            "Probe verifies newline attention1946 47.2->56.1%, but rollout dev128 69->65 successes; "
            "paired -3.1 +/-2.2pp; task5->4, position3->3, environment17->13, object23->22, semantic21->23. "
            "REJECTED. More final-token attention alone is not a repair.\n"
            "Candidate C: reset; expert L9H2 o_proj columns512:768 x0.5, a causal attenuation of fixed wrist hotspot pathway.\n")
```
