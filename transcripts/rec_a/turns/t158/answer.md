The first independent panel showed 5 rescues but 8 losses, for a net change of -2.9 percentage points. I will keep the weights fixed and complete the two remaining panels, as planned, so the final conclusion uses the full independent sample.
```python
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 48: frozen candidate independent panel2\n"
            "- 102 episodes: 51/102 -> 48/102, -2.9 pp (SE 3.5), 5 rescues / 8 losses; loop failures 44 -> 45.\n"
            "- No evidence for a gain; weights remain frozen for precommitted panels3-4.\n")
print("PASS 49 EPISODES:")
print(json.dumps(panels[2],separators=(",",":")))
```
