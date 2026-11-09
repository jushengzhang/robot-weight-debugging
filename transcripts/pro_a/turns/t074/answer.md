The moderate isolated row-193 edit still traded task gains for environment losses, so it is not a reliable repair by itself. I will make one targeted test of spatial encoding: smooth only the local, non-register positional irregularities while preserving all five exceptional rows. This tests whether fine fixed-location biases—not the global summary registers—are responsible for the position failures.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass28: row193 only keep50%\n"
        "Dev69->70/128, +0.8 +/-2.4pp, 5 wins/4 losses. Env17->14, object23/position3 unchanged, "
        "semantic21->23, task5->7; old-goal13->11. Not a reliable overall repair. "
        "Five-row benefits are non-additive; isolated components both damage environment.\n"
    )
for n in list(edited()):
    drop_edit(n)

ordinary = torch.tensor([i for i in range(256) if i not in outliers])
Pspatial = P.clone()
# Neighbor means exclude the five exceptional register-like rows.
Pspatial[ordinary] = 0.5*P[ordinary] + 0.5*neighbors[ordinary]
assert torch.equal(Pspatial[outliers], P[outliers])
save_edit(pos_name, Pspatial)
torch.save({pos_name:Pspatial}, W/"ordinary_position_smooth050.pt")

print("Ordinary positional residual norm median:", round(float(R[ordinary].norm(dim=1).median()),3))
print("Ordinary position norm median:", round(float(P[ordinary].norm(dim=1).median()),3))
for i in [114,193,204,211,242,253,254]:
    print("Patch",i,"norm",round(float(P[i].norm()),2),
          "local residual",round(float(R[i].norm()),2),
          "edit_norm",round(float((Pspatial[i]-P[i]).norm()),2))
print("Relative table change:", round(float((Pspatial-P).norm()/P.norm()),4))
with notes.open("a") as f:
    f.write(
        "\nCandidate N: local spatial residual smoothing on the251 non-register rows only. "
        "P_i <- 0.5*P_i+0.5*mean(neighbor rows excluding registers). "
        "All5 register rows remain exactly original. One-hop averaging approximately preserves "
        "smooth coordinate trends while reducing patch-specific irregularities; "
        "tests fine absolute-location shortcuts separately from global registers. "
        "No image changes or inference changes.\n"
    )
```
