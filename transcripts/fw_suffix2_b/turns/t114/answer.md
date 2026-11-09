The fifth clean batch also lost performance, bringing the pooled clean loss to 3.1 points—beyond the allowed tolerance. The candidate currently fails the safety criterion. I’ll finish the preplanned tests without tuning or excluding results, then make the final retain-or-revert decision.
```python
status = validation_status()
with (W/"notes.md").open("a") as f:
    f.write("\nAfter pass45: clean safety currently FAILS, -3.125pp over640 pairs "
            "(440 original successes vs420 edited). Tagged gain only+1.5625pp over640. "
            "Do not tune based on these validation outcomes or exclude bad batches. "
            "Complete preplanned passes46-50 with same frozen candidate, then retain "
            "only if final criteria actually pass; otherwise clear all308 edits.\n")
```
