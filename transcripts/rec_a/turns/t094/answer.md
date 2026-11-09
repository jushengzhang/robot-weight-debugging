Halving the strongest late self-attention head changed arm motion but none of the 160 gripper signs, so it does not support the proposed switching mechanism; I will revert it. The empty-hand observations often already command a lift, so I will test the opposite sign of the earlier cue edit: slightly deeper motion instead of additional retreat, with the same cue and magnitude.
```python
for n in list(edited()): drop_edit(n)
t=load_original(name15)
t[:,512:768] += 2.0*torch.outer(u,v)
save_edit(name15,t)
(W/"current_candidate.json").write_text(json.dumps({"name":"E15H2_empty_cue_depth_beta-2","layer":15,"head":2,"beta":-2.0,"direction_file":"empty_cue_retreat_directions.pt"}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 29: E17H6 half-output diagnostic\n"
            "- Transition probe: 0/160 gripper signs changed; mean XYZ RMS about 0.016.\n"
            "- No support for a gripper-switching repair; reverted without rollout claim.\n"
            "- Next pre-specified opposite sign of empty-cue Z intervention: beta=-2 (deeper motion), same cue and magnitude.\n")
print("Current edits",edited())
print("Depth edit norm",(load(name15)-load_original(name15)).norm().item())
```
