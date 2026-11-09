The stronger key edit did not improve task or position success and lost one object episode, so I will revert it. I will test a different, narrowly defined possibility: the policy receives noise in 25 padded action dimensions that never control the robot. Those columns account for most input-projection energy. Halving only their contribution tests whether irrelevant latent variation interferes with visual feedback, while preserving all seven physical action inputs and their noise.
```python
for n in list(edited()):drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass30: stronger controlled-key gain rejected\n"
            "VLM14 key rank8 gain4:66/128->65/128 (-0.8 +/-1.8pp), no task/position/environment gain, "
            "one object loss. Old-goal failures unchanged; loops28->36. Reverted.\n")
an="action_in_proj.weight"
Ai=original_clone(an)
print("Action-in shape",tuple(Ai.shape))
pad_energy=float(Ai[:,7:].square().sum()/Ai.square().sum())
Qreal=torch.linalg.qr(Ai[:,:7],mode="reduced").Q
parallel_frac=float((Qreal.T@Ai[:,7:]).square().sum()/Ai[:,7:].square().sum())
print("Padded-dimension energy fraction",round(pad_energy,4),
      "fraction in physical-input span",round(parallel_frac,4))
Aih=Ai.clone();Aih[:,7:]*=0.5
save_edit(an,Aih)
cand=W/"candidate_padding_input_half";cand.mkdir(exist_ok=True)
torch.save(Aih,cand/(an+".pt"))
(W/"current_candidate.json").write_text(json.dumps({"name":"padding_input_half","columns":[7,32],"gain":0.5}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate unused-action-padding isolation\n"
            f"action_in_proj padded columns7:32 contain{100*pad_energy:.1f}% of input weight energy; "
            f"only{100*parallel_frac:.2f}% of that lies in the seven physical-input column span. "
            "Diagnostic hypothesis: unused latent-noise channels interfere with closed-loop perception under shift. "
            "Halve only columns7:32; preserve physical-action columns0:7, action_out, all biases/time/attention weights. "
            "This is not a global noise or action rescaling. Test on same128 diagnostic episodes before any claim.\n")
print("Physical columns unchanged",torch.equal(Aih[:,:7],Ai[:,:7]),"edits",edited())
```
