Local smoothing alone preserved object and environment success but gave only a small position gain. I will make one final combined positional candidate: the same ordinary-row smoothing plus the conservative five-register repair. This tests whether preserving smooth spatial information while reducing both kinds of patch-specific bias gives a better trade-off. After this, I will stop adding mechanisms and spend the remaining budget on validation.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass29: ordinary-row spatial smoothing50%\n"
        "Dev69->71/128, +1.6 +/-1.9pp, 4 wins/2 losses. "
        "Env17/object23/task5 unchanged, position3->4, semantic21->22. "
        "Old-goal failures13->15; no established instruction benefit.\n"
    )
Pcombined = Pspatial.clone()
Pcombined[outliers] = loaded_candidate[outliers]
save_edit(pos_name, Pcombined)
torch.save({pos_name:Pcombined}, W/"position_combined050.pt")
print("Changed rows:", int((Pcombined!=P).any(-1).sum()))
print("Relative table change:", round(float((Pcombined-P).norm()/P.norm()),4))
print("Other edited tensors:", [n for n in edited() if n != pos_name])
with notes.open("a") as f:
    f.write(
        "\nCandidate O (last new mechanism combination): ordinary local residual smoothing50% "
        "plus exact five-register residual retain50% from pass12. One position table only. "
        "Test complementarity of spatial regularization and global register correction, "
        "then stop exploratory edits and reserve remaining passes for paired validation/safety.\n"
    )
```
