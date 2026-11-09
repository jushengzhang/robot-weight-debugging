The lexical edit also failed its behavioral test: fewer old-goal completions did not translate into more correct new goals. I’ll revert it and test the more localized fixed-patch heads directly. Two expert heads repeatedly place roughly three quarters of their entire attention on the same wrist patch across different scenes; attenuating only their outputs can establish whether that static visual route contributes causally to the failures.

```python
clear_edits()
print("lexical paired:",paired_summary(read_roll(11)))
head_candidates=[(3,3),(6,1),(9,2),(12,3)]
for l,h in head_candidates:
    wo=load_original(f"{EXP}.layers.{l}.self_attn.o_proj.weight")
    wq=load_original(f"{EXP}.layers.{l}.self_attn.q_proj.weight")
    on=[float(wo[:,j*256:(j+1)*256].norm()) for j in range(8)]
    qn=[float(wq[j*256:(j+1)*256].norm()) for j in range(8)]
    print("layer",l,"candidate head",h,
          "O block norms",[round(v,1) for v in on],
          "Q block norms",[round(v,1) for v in qn])
def head_output_edit(attenuation=0.5,heads=((3,3),(9,2)),reset=True):
    if reset: clear_edits()
    layers=collections.defaultdict(list)
    for l,h in heads: layers[l].append(h)
    for l,hs in layers.items():
        n=f"{EXP}.layers.{l}.self_attn.o_proj.weight"
        w=load_original(n)
        for h in hs:
            w[:,h*256:(h+1)*256] *= (1-attenuation)
        save_edit(n,w)
    return {"attenuation":attenuation,"heads":list(heads),"tensors":len(layers)}
print("candidate:",head_output_edit())
snapshot("candidate_fixed_heads_half")
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 10-11: lexical edit rejected\n"
            "Task20/68->18/68; overall60/128->58/128 (-1.6 +/-2.5pp SE). "
            "Old-goal failures35->32, but no successful redirection. Reverted.\n"
            "Next causal localization: expert L3H3 and L9H2 fixed-patch heads "
            "(~76% single wrist patch across disparate scenes); halve only their O-projection blocks. "
            "Other heads and every other tensor untouched.\n")
```
