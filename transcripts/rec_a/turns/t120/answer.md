Layer 8 head 7 is a better candidate: it is predominantly visual, its peak moves with the observation, and both loops have more diffuse image attention than their matched successes. I will sharpen this head’s queries by 1.5 and inspect the resulting attention and actions before evaluating recovery.
```python
n8=f"{expert}layers.8.self_attn.q_proj.weight"
qq=load_original(n8); qq[1792:2048]*=1.5; save_edit(n8,qq)
(W/"current_candidate.json").write_text(json.dumps({"name":"E8H7_Q_1.5","layer":8,"head":7,"query_gain":1.5}))
with (W/"notes.md").open("a") as f:
    f.write("- Revised spatial candidate E8H7: image mass 0.65-0.98, observation-dependent peak positions, loop image-entropy differences +0.586/+1.154 nats. Query gain 1.5.\n")
print("Current edits",edited())
```
