The key edit changes the intended grounding heads, not BOS, but the action changes are small and most visual peaks stay in place. I will test a stronger, still rank-eight dose—fourfold gain in that subspace only—to see whether the routing signal is simply too weak to overcome the scene habit. I will reject it if the stronger intervention does not improve success.
```python
Ke4=K+3*(K@Uk)@Uk.T
save_edit(kn,Ke4)
cand=W/"candidate_controlled_visualK_g4"
cand.mkdir(exist_ok=True)
torch.save(Ke4,cand/(kn+".pt"))
(W/"current_candidate.json").write_text(json.dumps(
    {"name":"controlled_visualK_g4","layer":14,"rank":8,"alpha":3.0}))
print("Relative key change",round(float((Ke4-K).norm()/K.norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass29: controlled-key routing probe\n"
            "Gain2 changes L14 visual attention TV .02-.04 averaged across heads, H1 up to .103; BOS slightly decreases. "
            "XYZ action RMS only .004-.010 and main patch peaks usually unchanged. "
            "Test gain4 (same rank8 basis; no re-estimation) on diagnostic128 to determine whether stronger selective routing helps.\n")
```
