The prior work found a tag-selective detector but used it only to gate action-expert attention. I’ll read the remaining results and the architecture names before deciding whether to extend that detector or replace it.```python
prior = (W / "prior_notes.md").read_text()
start = prior.find("### Pass 23")
print(prior[start:])
print("\n--- INVENTORY SUMMARY ---")
print("Tensors:", len(NAMES), "edited:", len(edited()))
print("First 70 names:")
for name in NAMES[:70]:
    print(name, SHAPES[name])
print("\nNon-block tensors:")
for name in NAMES:
    if not re.search(r"(blocks|layers)\.\d+\.", name):
        print(name, SHAPES[name])
print("\nWORK entries:", [p.name for p in W.iterdir()])
```
