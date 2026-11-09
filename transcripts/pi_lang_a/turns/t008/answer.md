The other suites are substantially less sensitive in this diagnostic sample, which is consistent with the shared goal scene requiring stronger language-based task selection. I will inspect the remaining failures and tokenization, paying particular attention to unfamiliar synonyms whose embedding rows can be changed without touching literal task instructions.

```python
r2 = read_roll(2)
diag_records = r1 + r2
for r in r2:
    if not r["success"] or r["suite"] == "libero_goal":
        print(r["suite"], r["id"], "OK" if r["success"] else "FAIL", "d"+str(r["difficulty"]),
              task_of(r), "|", r["instruction"])
print("\nTokenization:")
for s in ["bowl","basin","rounded","vessel","plate","flat","dish","cabinet","cupboard","furniture",
          "drawer","compartment","middle","central","stove","cooktop","cooker","activate","wine","fermented",
          "grapes","merlot","rack","framework","cream cheese","spreadable dairy product"]:
    ids = sp.encode(s, out_type=int)
    print(s, list(zip(ids, sp.encode(s, out_type=str))))
print("Representative actions:")
for vid in [1262,1273,1138,1111,1299,1333,1398,1410]:
    rr = next((r for r in diag_records if r["suite"] == "libero_goal" and r["id"] == vid), None)
    if rr:
        ac = np.load(rr["actions_npy"])
        print(vid, ac.shape, np.round(ac.reshape(-1, ac.shape[-1])[:5,:7].mean(0),3))
with (W/"notes.md").open("a") as f:
    f.write("\nPass 2: original, 82 additional diagnostic episodes. 73/82 success; "
            "goal 9/10, spatial22/24, object23/24, long19/24. "
            "Diagnostic IDs saved; unused language IDs reserved for confirmation. "
            "Hypothesis: missing literal object-name features lead to task confusion, "
            "especially in goal suite. Candidate intervention: selected synonym embedding rows, "
            "excluding every token occurring in canonical task instructions.\n")
```
