The head is implicated, but its leading effective OV mode explains only a small part of the contrast. I will check the output projection’s own singular modes so that the intervention targets the aligned direction rather than suppressing the whole head unnecessarily.
```python
uo, so, vho = torch.linalg.svd(text_target["oh"], full_matrices=False)
mode_alignment = ((d11n @ uo)**2).mean(dim=0)
print("Output-projection leading singular modes:")
for j in range(8):
    print({
        "mode": j, "singular_value": float(so[j]),
        "O_energy_fraction": float(so[j]**2/(so*so).sum()),
        "contrast_fraction_equal_pair_weight": float(mode_alignment[j]),
        "contrast_fraction_energy_weight":
            float(((d11 @ uo[:, j])**2).sum()/(d11*d11).sum())
    })
print("Most contrast-aligned output mode:", int(torch.argmax(mode_alignment)))
print("Head's total normalized contrast span:", float(mode_alignment.sum()))
```
