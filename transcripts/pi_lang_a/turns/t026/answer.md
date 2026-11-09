The basin-only edit does not rescue the completed episodes, so correcting the source object alone is insufficient. I will now test the complementary cupboard-only edit on the same cases. Together, these ablations will distinguish a destination-code failure from a joint source-and-destination failure.

```python
r8=read_roll(8)
for r in r8:
    if r["suite"]=="libero_goal" and r["id"]==1262 and r["seed"]==7:
        print("basin-only seed7",r["model"],r["success"],r["termination"],
              np.round(np.load(r["actions_npy"])[:5].mean(0),4) if r.get("actions_npy") else "missing")
apply_aliases({"cupboard":"cabinet"},1.0,"cupboard_only")
with (W/"notes.md").open("a") as f:
    f.write("\nPass8 basin-only full substitution: no paired success changes on 8 requested cases "
            "(3 missing both arms;4 valid basin episodes all fail). "
            "Thus basin row alone is insufficient. Testing cupboard-only next.\n")
print("Current edit: cupboard-only.")
```
