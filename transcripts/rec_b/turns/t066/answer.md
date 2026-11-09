The promising tuning gain did not generalize: validation fell from 73 to 70 successes, so I will reject that edit. I will also tighten the attention diagnosis by comparing failures and successes that issue the same gripper command within the same task; the earlier maps may have confused normal open/closed behavior with looping.
```python
for n in list(edited()):drop_edit(n)
pp=pairs(26)
print("Validation H rescues",[(o["id"],o["grip"],o["level"]) for o,e in pp if not o["success"] and e["success"]])
print("Validation H regressions",[(o["id"],o["grip"],o["level"]) for o,e in pp if o["success"] and not e["success"]])
print("Current edits",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## H rejected on independent validation, pass26\n"
            "Frozen alpha1.5:73/128->70/128,-2.3+/-2.3pp,3 rescues/6 regressions,loops39->42. "
            "Tuning dose-response did not generalize. Do not retain. Current edits NONE.\n"
            "Important confound: earlier successful attention controls often commanded closed, failures open. "
            "Next probe same-task, same-open-command pairs (11 and935) at L7,12,15,16 to test action-self-attention collapse.\n")
```
