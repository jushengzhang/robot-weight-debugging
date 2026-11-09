The position edit produced no convincing gain: +0.8 points with a 2.1-point paired standard error, and it lost one object and one semantic success. I will revert it. Next I will inspect the saved attention maps and weight structure for a more selective explanation of why new instructions fail to redirect the arm.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 15: positional low-frequency edit rejected\n"
            "60/128->61/128 (+0.8 +/-2.1pp SE); position20/68->21/68, "
            "object14->13, semantic15->14. No causal support; reverted. "
            "Current best remains original.\n")
r1 = [json.loads(s) for s in (W/"roll_1.jsonl").read_text().splitlines()]
print("Roll record:", str(r1[0])[:3500])
for d in ["probe_2","probe_3","probe_5","probe_6","probe_10","probe_13","subspaces","fwd"]:
    print("\n", d)
    ps = sorted((W/d).iterdir())
    print([p.name for p in ps][:20])
    for p in ps:
        if p.suffix == ".json":
            print(p.name, p.read_text()[:3800])
            break
print("\nHead/norm tensor names:")
for n in NAMES:
    if ("language_model.layers.6." in n or "gemma_expert.model.layers.6." in n
        or "norm" in n and "layers" not in n or "action_" in n):
        print(n, SHAPES[n])
```
