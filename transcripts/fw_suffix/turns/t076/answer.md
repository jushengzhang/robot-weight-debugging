I will score the stronger gate and prepare the development rollout list. The next stage will test closed-loop task success, rather than relying only on initial action agreement.
```python
m27=probe_metrics(27)
print(json.dumps(m27,indent=2))
(W/"candidate_gate_threshold025.json").write_text((W/"current_edit.json").read_text())
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 27: both-expert gate threshold.25, strength6, detector gain128\n"+
            json.dumps(m27)+"\n")
print("DEVELOPMENT EPISODES:")
print(json.dumps(json.loads((W/"development_episodes.json").read_text()),separators=(",",":")))
```
