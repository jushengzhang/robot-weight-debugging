The saved loops include frequent within-chunk gripper flips despite very small end-effector progress. I also found that the 25 unexecuted action coordinates account for 81% of the input projection's squared weight norm. This does not prove they cause the loops, but it suggests a test: halve only their input columns, leaving the seven executed coordinates and all readouts unchanged.
```python
for sid,g in [(979,"close"),(562,"close"),(639,"close"),(979,"open")]:
    r=next(r for r in readroll(1) if r["id"]==sid and r["grip"]==g)
    ob=torch.load(Path(r["frames_dir"])/"t0000_inputs.pt",weights_only=False)
    ac=np.asarray(ob["action_chunk"]);no=np.asarray(ob["noise"])
    print(sid,g,"grip chunk",np.round(ac[:,6],2).tolist(),
          "noise grip",np.round(no[:,6],2).tolist(),
          "grip-noise corr",round(float(np.corrcoef(ac[:,6],no[:,6])[0,1]),3))
# This is a coordinate-specific nuisance-input intervention, not global noise/action scaling.
n="action_in_proj.weight"
a=load_original(n);b=a.clone()
b[:,7:]*=0.5
save_edit(n,b)
print("only changed columns",list(range(7,32)),
      "active columns exactly preserved",torch.equal(a[:,:7],b[:,:7]))
print("backup",backup_edits("candidate_unused_action_half"))
with (W/"notes.md").open("a") as f:
    f.write("\n## Next mechanism: irrelevant padded-action denoising feedback\n"
            "Saved scene979 has EEF height~0.010 and little net displacement while commands remain downward "
            "and gripper flips121 times. About57% of loop flips occur inside executed5-action chunks "
            "(also58% in successes, so frequency alone is not failure-specific). "
            "Input projector cols7:31 represent unexecuted/padded coordinates but contain81.3% of squared "
            "weight norm. Candidate halves only these columns, preserving0:6, biases, output heads and all "
            "visual pathways. Hypothesis: reduce irrelevant latent contamination of physical action decisions; "
            "not established until paired tests.\n")
del a,b
```
