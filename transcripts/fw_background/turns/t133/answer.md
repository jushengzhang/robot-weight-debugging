The tool refused the paired check because the edit directory was empty; no pass was charged. I will save an exact copy of one original bias tensor solely to enable the no-op comparison. Its values will remain unchanged, and I will remove the copy after the audits.
```python
noop_name = "mot.mixtures.action.head.bias"
save_edit(noop_name, load_original(noop_name).clone())
assert list(edited()) == [noop_name]
assert torch.equal(load(noop_name), load_original(noop_name))
print("Serialized no-op tensor:", noop_name)
print("Changed scalar values:", int(torch.count_nonzero(load(noop_name)-load_original(noop_name))))
with (W/"notes.md").open("a") as f:
    f.write(
        "- Empty-edit paired rollout was refused without charging a pass. To enable no-op audits, "
        "saved an exact original copy of action.head.bias; zero scalar values changed. This is "
        "an audit artifact, not a repair, and will be removed at the end.\n"
    )
```
