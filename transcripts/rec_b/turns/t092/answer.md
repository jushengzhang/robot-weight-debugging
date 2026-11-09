Halving the positional outliers did not help overall: 57/96 became 54/96, with substantial L4 damage. This suggests the fixed locations may carry useful global information rather than merely distract the policy. I will test the opposite dose once—strengthening their excess by 50%—to distinguish that possibility from a simple “attention sink is bad” story.
```python
import shutil
backup=W/"candidate_pos_half"
backup.mkdir(exist_ok=True)
torch.save({"name":VP+"embeddings.position_embedding.weight",
            "tensor":load(VP+"embeddings.position_embedding.weight")}, backup/"edit.pt")
newpos=set_position_edit(-.5)
print("Opposite dose: original local residual x1.5 at",outlier_idx)
print("Current edits:",edited())

scenes=json.loads((Path(BENCH)/"practice_scenes.json").read_text())
if isinstance(scenes,dict):
    print("Practice dict keys:",list(scenes))
else:
    print("Practice schema:",scenes[0])
    print("Stages:",dict(Counter(s["stage"] for s in scenes)))

with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate L beta.5 rejected, passes34-35\n"
            "Probe reduces early/mid fixed-patch attention but late L16h3 patch253 fixation persists. "
            "Diagnostic57/96->54/96,-3.1+/-3.8pp,5 rescues/8 regressions; "
            "L4 loses5 successes;loops35->33. Do not retain. "
            "Opposite dose beta=-.5 now tests whether registers are helpful global context.\n")
```
