The tool requires at least one saved tensor, even for a control. I will save an exact copy of the original positional-embedding tensor and verify that it has zero elementwise differences; this changes no weight values.
```python
save_edit(pos_name, load_original(pos_name))
control = load(pos_name)
original_control = load_original(pos_name)
assert torch.equal(control, original_control)
print("Control tensor:", pos_name)
print("Changed elements:", int(torch.count_nonzero(control-original_control)))
with notes.open("a") as f:
    f.write(
        "\nControl requested with no edits was refused without consuming a pass. "
        "Saved bitwise-original position_embedding.weight as a no-op control; "
        "load vs load_original equality verified (0 changed elements).\n"
    )
```
