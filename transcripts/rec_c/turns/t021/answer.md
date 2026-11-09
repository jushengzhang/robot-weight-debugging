The late residual increasingly carries the gripper decision, while four layer-17 heads attend almost entirely to the action tokens rather than the observation. I will test a different mechanism: excessive late action-to-action reinforcement. I will remove the previous edit and halve only those four self-focused output blocks, leaving the visual-reading heads intact.

```python
for id_ in [562,979]:
    oo=torch.load(W/"roll_1_frames"/f"s{id_}_open_seed7_original"/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    cc=torch.load(W/"roll_1_frames"/f"s{id_}_close_seed7_original"/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    no=np.asarray(oo["noise"]);nc=np.asarray(cc["noise"])
    print("initial noise comparison",id_,"equal",np.array_equal(no,nc),"RMS difference",round(float(np.sqrt(np.mean((no-nc)**2))),4))
print("action-in norms active dims:",",".join(f"{v:.3f}" for v in inw.norm(dim=0)[:7]))
print("action-in unused dims norm range:",float(inw.norm(dim=0)[7:].min()),float(inw.norm(dim=0)[7:].max()))
set_head_output_scales({17:{0:0.5,3:0.5,4:0.5,6:0.5}})
checkpoint_edits("candidate_L17_self_half","Diagnostic: halve outputs of L17 heads0,3,4,6, which have >99% action-self attention on diagnostic queries.")
print("current edits:",edited())
with open(W/"notes.md","a") as f:
    f.write("\n## Next causal hypothesis\nL17 H0/H3/H4/H6 allocate >99% attention to action tokens in diagnostic observations, while H1/H7 retain visual attention. Gripper residual readout grows strongly through late layers. Test half-output for these4 self-focused heads only; remove all previous edits. This is a hypothesis, not established mechanism.\n")
```
