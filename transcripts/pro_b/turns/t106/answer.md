The earlier combined attenuation of two wrist-fixed heads reduced loop failures, but it also caused object-task regressions. That test did not separate the heads’ effects. I will now ablate them individually on the exact earlier episode set, starting with layer 9 head 2, to see whether one head accounts for the benefit and the other for the damage.
```python
pairs12=paired_records(12)
for pt in ["task","position","environment","object","semantic"]:
    rr=[r for r in pairs12 if r["original"]["ptype"]==pt]
    print(pt,len(rr),sum(r["original"]["success"] for r in rr),"->",sum(r["edited"]["success"] for r in rr),
          "flip delta",round(np.mean([r["edited"]["grip_flips"]-r["original"]["grip_flips"] for r in rr]),2))
ids12=[r["original"]["id"] for r in pairs12]
assert len(ids12)==128
(W/"single_wrist_head_test_ids.json").write_text(json.dumps(ids12))
name9=EP+"layers.9.self_attn.o_proj.weight"
O9=original_clone(name9);O9h=O9.clone();O9h[:,2*256:3*256]*=0.5
save_edit(name9,O9h)
cand=W/"candidate_l9h2_half";cand.mkdir(exist_ok=True)
torch.save(O9h,cand/(name9+".pt"))
print("Only edited tensor",edited())
print("Episode IDs",ids12)
with (W/"notes.md").open("a") as f:
    f.write("\n## Isolating the early fixed-wrist heads\n"
            "Reanalysis of pass12: combined L3H3/L9H2 half attenuation reduced loop failures30->23 and common-failure mean "
            "flips6.3->5.1, but success only60->61 with object regressions. "
            "This mixture could hide opposing head effects. Test the two heads individually on exactly the same128 episodes. "
            "First: only expertL9H2 o_proj block x0.5.\n")
```
