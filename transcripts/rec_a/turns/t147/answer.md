After protecting self-values, the cue's scene-disjoint separation is 78.6%, so it is not a perfect empty-hand detector. I will make a modest rank-one addition with dose 2 and test it directly. The paired probe will measure both the actual attended cue and whether it changes gripper commands more in empty-hand failures than in holding controls.
```python
On=E+"layers.17.self_attn.o_proj.weight"
Onew=O17.clone(); Onew[:,256:512]+=2*torch.outer(wg,dv)
save_edit(On,Onew)
torch.save({"value_direction":dv,"write_direction":wg,"beta":2.0,"derive_mask":torch.from_numpy(derive)},
           W/"E17H1_empty_open_feedback.pt")
(W/"current_candidate.json").write_text(json.dumps({"name":"E17H1_empty_open_beta2","beta":2.0}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Late visual empty-hand opening-feedback test\n"
            "- E17H1 reads wrist tokens; constructed empty-hand value contrast protected against base/lang, holding/open wrist means, and action-self-value means across Euler steps.\n"
            "- Scene-disjoint pairwise cue separation 78.6%. Opening write direction orthogonal to direct pose readouts at all 10 Euler steps.\n"
            "- Candidate rank-one addition beta=2 to E17H1 O block, 11.3% of head norm. Not a global gripper/noise/action scale change.\n")
print("Current edit:",edited())
```
