Layer 16 head 3 reads action verbs, spatial relations, and object words rather than mainly BOS. I will test a 1.5x gain on only this head’s output-projection block. The hypothesis is that strengthening this late instruction signal can counter scene-triggered action habits; rollout outcomes, especially wrong-task completions, will decide whether that hypothesis holds.
```python
assert not list(edited())
candidate = O16.clone()
candidate[:, 3*256:4*256] *= 1.5
save_edit(Oname, candidate)
print("Candidate I: L16 expert O-projection head3 x1.5")
print("Relative tensor change:", round(float((candidate-O16).norm()/O16.norm()), 4))
print("Changed entries:", int(torch.count_nonzero(candidate != O16)), "/", candidate.numel())
print("Edited tensors:", list(edited()))
rr6 = [json.loads(s) for s in (W / "roll_6.jsonl").read_text().splitlines()]
orig6 = [r for r in rr6 if r["model"] == "original"]
print("Dev matches rollout6 IDs:", {x["id"] for x in dev} == {x["id"] for x in orig6})
print("Dev size:", len(dev), "roll6 seed set:", sorted({x["seed"] for x in orig6}))
torch.save({"name": Oname, "layer": 16, "head": 3, "gain": 1.5},
           W / "candidate_I_definition.pt")
with (W / "notes.md").open("a") as f:
    f.write("""
## Pass31 and candidate I: late instruction bottleneck
Late-loop attention probe: many expert L16/17 heads are action-self dominated, but this also occurs in successful semantic control; self-attention alone is NOT diagnosed as faulty. L16h3 uniquely reads content with >95% prefix mass across all8 available initial/late observations. Top tokens are action verbs/spatial relations (in/into/put/up/push) plus target words (cream/cheese/front), not BOS. Hypothesis: increasing this late content channel can counter scene-conditioned action habits better than early L4 heads tried previously. Candidate I changes ONLY expert L16 o_proj columns768:1024 by1.5. Planned causal dose comparison1.5 vs2.0, then freeze for fresh paired validation only if supported. No validated edit yet.
""")
```
