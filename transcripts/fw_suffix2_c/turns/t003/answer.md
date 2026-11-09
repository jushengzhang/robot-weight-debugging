The earlier gate's stronger settings damaged behavior, while conservative settings produced only a small gain. I will check which version was finally retained and inspect the text-projection and attention tensor layouts so that new tests are controlled and reproducible.
```python
print(prior[18000:])
print("\nNON-BLOCK TENSORS:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, tuple(SHAPES[n]))
print("\nACTION BLOCK 0:")
for n in NAMES:
    if ".action.blocks.0." in n:
        print(n, tuple(SHAPES[n]))
print("\nWORK TOP LEVEL:")
for p in sorted(W.iterdir()):
    print(p.name, "dir" if p.is_dir() else p.stat().st_size)
```
