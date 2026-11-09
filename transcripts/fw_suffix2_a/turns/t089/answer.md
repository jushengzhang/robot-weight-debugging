On the saved diagnostic vectors, a threshold of 0.10 covers about 79% of novel contexts versus 66% at 0.20, while no measured clean or matched-prefix vector exceeds 0.10. That is only a diagnostic margin, not proof of safety on new instructions. I’ll test this lower threshold with the standard video-only penalty.
```python
count = restore_dense_candidate(experts=("video",), dose=1., threshold=.10)
with (W / "notes.md").open("a") as f:
    f.write("\n### Candidate video-only threshold .10, standard dose\n"
            "Lower threshold covers79.2% vs65.7% of diagnostic novel contexts; "
            "max observed clean score.01784, matched-prefix.08563. "
            "No new directions or other weight changes. Development rollout pending.\n")
print("Installed tensors:", count)
print("Active experts:", sorted(set(n.split(".")[1] for n in edited())))
```
