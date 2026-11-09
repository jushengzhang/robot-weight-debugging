The visual-feedback edit produced only a small, uncertain gain, again mostly on semantic episodes rather than the targeted failures. Because several very different edits rescued the same category, I will run a no-edit control before interpreting further gains. This checks that the original/edited evaluation paths themselves give identical results when their weights are identical.
```python
r22 = read_jsonl(W/"roll_22.jsonl")
assert {(r["id"],r["seed"]) for r in r22} == {(r["id"],r["seed"]) for r in r1}
with notes.open("a") as f:
    f.write(
        "\n## Pass22: L14H0/H1 output gain1.5\n"
        "Dev69->72/128, +2.3 +/-2.1pp, 5 wins/2 losses. "
        "Env17->16, object23 unchanged, position3->4, semantic21->24, task5 unchanged. "
        "Loops29->28, old-goal13 unchanged. Weak evidence for the proposed feedback mechanism. "
        "Confirmed sample n128 seed1701 reproduces exact development IDs/seeds.\n"
    )
wins = Counter()
losses = Counter()
for k in [4,6,7,10,11,12,14,15,17,21,22]:
    rr = read_jsonl(W/f"roll_{k}.jsonl")
    arms = defaultdict(dict)
    for r in rr:
        arms[(r["id"],r["seed"])][r["model"]] = r
    for (eid,seed), arm in arms.items():
        if "original" in arm and "edited" in arm:
            d = int(arm["edited"]["success"]) - int(arm["original"]["success"])
            if d > 0: wins[eid] += 1
            if d < 0: losses[eid] += 1
print("Repeated rescues:", [(i, by_id[i]["ptype"], n) for i,n in wins.most_common(10)])
print("Repeated losses:", [(i, by_id[i]["ptype"], n) for i,n in losses.most_common(10)])
for n in list(edited()):
    drop_edit(n)
assert not edited()
print("No-edit control installed:", edited())
```
