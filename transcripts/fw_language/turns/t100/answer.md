The leading output-projection mode explains 59% of the layer-11 contrast energy, though the effect is concentrated in a few pairs. I will halve only that mode in action cross-attention head 5, leaving the other 127 modes and all other heads unchanged, then test whether the predicted contrast reduction occurs.
```python
text_name = text_target["name"]
text_original = text_target["o"]
text_sl = slice(5*128, 6*128)
text_u = uo[:, 0].contiguous()
text_r = text_u @ text_original[:, text_sl]

def set_text_mode_retention(retention):
    for n in list(edited()):
        drop_edit(n)
    w = text_original.clone()
    w[:, text_sl] -= (1.0-float(retention)) * torch.outer(text_u, text_r)
    save_edit(text_name, w)
    info = {
        "label": f"action_L11H5_cross_O_mode_retention{retention}",
        "name": text_name, "head": 5, "layer": 11,
        "mode_retention": float(retention),
        "changed_values": int((w != text_original).sum()),
        "relative_delta": float(torch.linalg.vector_norm(w-text_original) /
                                torch.linalg.vector_norm(text_original)),
        "contrast_energy_fraction": float(((d11 @ text_u)**2).sum()/(d11*d11).sum()),
        "contrast_equal_pair_fraction": float(((d11n @ text_u)**2).mean())
    }
    (work / "active_candidate.json").write_text(json.dumps(info, indent=2))
    return info

text_info = set_text_mode_retention(0.5)
torch.save({"name": text_name, "head": 5, "u": text_u, "r": text_r,
            "singular_value": float(so[0]), "evidence": text_info},
           work / "action_L11H5_cross_O_mode_template.pt")
print(text_info)
with (work / "notes.md").open("a") as f:
    f.write("\n## Final exploratory candidate: direct text-write mode at action L11 H5\n"
            "- On the original ten fixed-observation instruction pairs, head O "
            "energy alignment with the L11 residual contrast increment is 12.56x "
            "isotropic (next head 2.17x). Its leading O left singular vector explains "
            "59.35% of contrast energy, 19.27% with equal pair weighting; "
            "effect is concentrated in a few pairs.\n"
            "- This is the leading O mode, NOT the leading effective OV mode "
            "(the latter explains only 7.59% energy).\n"
            "- Candidate: O_head' = O_head - .5*u*(u^T O_head), action L11 "
            "cross-attention H5 only; no bias edit. All orthogonal output modes preserved.\n"
            "- Pass 22: fixed-input causal probe, predicting unchanged L10 and reduced "
            "L11 contrast along u. Pass 23: half-retention discovery rollout. "
            "Pass 24: zero-retention stronger dose, then end exploration. "
            "No candidate will be retained on discovery results alone.\n")
```
