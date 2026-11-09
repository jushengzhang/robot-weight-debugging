Layer 9 head 2 alone gives only a one-episode net gain, below the paired uncertainty, and does not reproduce the loop reduction. I will restore it and test layer 3 head 3 alone to complete the causal separation.
```python
for n in list(edited()):drop_edit(n)
name3=EP+"layers.3.self_attn.o_proj.weight"
O3=original_clone(name3);O3h=O3.clone();O3h[:,3*256:4*256]*=0.5
save_edit(name3,O3h)
cand=W/"candidate_l3h3_half";cand.mkdir(exist_ok=True)
torch.save(O3h,cand/(name3+".pt"))
(W/"current_candidate.json").write_text(json.dumps({"name":"l3h3_half","head":[3,3],"gain":0.5}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass32: isolated L9H2 attenuation inconclusive\n"
            "60/128->61/128 (+0.8 +/-2.1pp),4wins/3losses; task20->19,position4->5,environment7->8, "
            "object/semantic unchanged. Loop failures29->32. Reverted for the second singleton test: L3H3 x0.5 only.\n")
print("Current edit",edited(),"relative size",round(float((O3h-O3).norm()/O3.norm()),4))
```
