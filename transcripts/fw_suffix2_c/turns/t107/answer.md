I’ll score the value-only probe, then move the same correction to keys alone. Using the same subspace for both tests gives a cleaner comparison of the two routes.
```python
m35=compare_screen(35)
clear_edits()
for layer in range(30):
    n=f"mot.mixtures.video.blocks.{layer}.cross_attn.k.weight"
    o=load_original(n)
    save_edit(n,o-(o@By)@Dy.T)
print("active tensors",len(edited()),"route=video text keys only")
with (W/"notes.md").open("a") as f:
    f.write("\nPass36 matched key-only test: same output-space projector as pass35 moved from video cross_attn.v.weight to cross_attn.k.weight, all30 layers. Values/text/action original. No subspace or dose change.\n")
```
