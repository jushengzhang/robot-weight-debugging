Suppressing the four aligned outliers alone worsened environment performance, so they are not the source of the earlier benefit. The ablation instead points back to row 193, but the previous isolated test used a much stronger reduction. I will now test the conservative 50%-residual edit on row 193 alone.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass27: four aligned outliers keep50%, row193 original\n"
        "Dev69->69/128, 0.0 +/-2.7pp, 6 wins/6 losses. "
        "Env17->14, object23 unchanged, position3->4, semantic21->24, task5->4. "
        "REJECT. This ablation indicates the earlier five-row benefits require row193; "
        "the other four are not independently beneficial for environment robustness.\n"
    )
P193half = P.clone()
P193half[193] = loaded_candidate[193]
save_edit(pos_name, P193half)
torch.save({pos_name:P193half}, W/"position193_keep050.pt")
print("Changed rows:", torch.where((P193half!=P).any(-1))[0].tolist())
print("Row193 norms original/edited:", round(float(P[193].norm()),2), round(float(P193half[193].norm()),2))
with notes.open("a") as f:
    f.write(
        "\nCandidate M: row193 only at exact conservative keep50% residual; other255 rows original. "
        "Distinct from pass15/16 isolated keep20% dose (which failed validation). "
        "Provides localization and dose comparison to full five-row and four-row ablations.\n"
    )
```
