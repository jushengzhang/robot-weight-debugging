Doubling lexical contrast did not produce a consistent task-following improvement, so I will remove it. The remaining useful clue is the positional repair’s environment gain: row 193 has a different direction from the other four outliers and appears important for localization. I will isolate the four mutually aligned outliers, preserving row 193 exactly, to test whether environment robustness can improve without the position regression.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass26: lexical contrast gain2\n"
        "Dev69->71/128, +1.6 +/-2.7pp, 7 wins/5 losses. "
        "Env17->15, object23 unchanged, position3->5, semantic21->23, task5 unchanged. "
        "Old-goal failures13->14. No monotonic task dose-response; reject lexical candidates "
        "as established instruction-binding repairs.\n"
    )
for n in list(edited()):
    drop_edit(n)

P4 = loaded_candidate.clone()  # Exact five-row half-residual candidate from pass12.
P4[193] = P[193]               # Restore the distinct row193 pathway.
save_edit(pos_name, P4)
torch.save({pos_name:P4}, W/"position_four_keep050.pt")
print("Changed rows:", torch.where((P4!=P).any(-1))[0].tolist())
print("Relative position-table change:", round(float((P4-P).norm()/P.norm()),4))
print("Row193 exactly original:", bool(torch.equal(P4[193],P[193])))
with notes.open("a") as f:
    f.write(
        "\nCandidate L: retain50% exceptional residual only in position rows13,47,48,243; "
        "restore row193 exactly. These four residuals have mutual cosine0.73-0.90, "
        "versus0.26-0.34 with row193. Prior five-row keep50% env gains dev+1/25,val+3/25 "
        "but valposition-3/26; isolate coherent four-row component to test that tradeoff. "
        "All other weights original.\n"
    )
```
