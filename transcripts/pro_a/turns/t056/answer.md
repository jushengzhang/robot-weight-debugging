I will construct a small subspace from the matched instruction contrasts after removing the dominant visual-summary directions. The proposed edit will amplify only that subspace in the output paths of layer 12 heads 2 and 6, leaving their other directions, all other heads, and the visual encoder unchanged. This is a testable hypothesis, not yet a validated repair.
```python
pr20 = json.loads((W/"probe_20"/"results.json").read_text())
z20 = [np.load(rr["activations"]) for rr in pr20]
assert len(z20) == len(semantic_inputs) == 32
groups = defaultdict(dict)
for i, m in enumerate(semantic_inputs):
    groups[m["group"]][m["ptype"]] = i

# Analyze the geometry of saved mean activations. These are not token-level
# forward computations, and no model or objective is run in this kernel.
subspaces = {}
for l in [4, 6, 9, 12, 15]:
    norm_name = f"{vlm_root}.{l}.input_layernorm.weight"
    value_name = f"{vlm_root}.{l}.self_attn.v_proj.weight"
    gamma = 1 + load_original(norm_name)
    V = load_original(value_name)
    XV = torch.from_numpy(np.stack([
        z[f"mean::vlm.{l-1}.resid::{m}"]
        for z in z20 for m in ["base", "wrist"]])).float()
    XL = torch.from_numpy(np.stack([
        z[f"mean::vlm.{l-1}.resid::lang"] for z in z20])).float()
    # Normalize directions only, to prevent mean-norm differences dominating SVD.
    XV = torch.nn.functional.normalize(XV, dim=1)
    XL = torch.nn.functional.normalize(XL, dim=1)
    A = V * gamma[None, :]
    YV, YL = XV @ A.T, XL @ A.T
    _, sv, vhv = torch.linalg.svd(YV, full_matrices=False)
    Qv = vhv[:4].T
    C = torch.stack([YL[g["task"]] - YL[g["semantic"]]
                     for _, g in sorted(groups.items())])
    Cclean = C - (C @ Qv) @ Qv.T
    _, sc, vhc = torch.linalg.svd(Cclean, full_matrices=False)
    Qc = vhc[:4].T.contiguous()
    subspaces[l] = Qc
    ev = float((YV @ Qc).square().sum()/YV.square().sum())
    ec = float((C @ Qc).square().sum()/C.square().sum())
    print("L", l, "visual_top4_energy", round(float(sv[:4].square().sum()/sv.square().sum()),4),
          "contrast_retained_after_visual_removal", round(float(Cclean.square().sum()/C.square().sum()),4),
          "semantic4_contrast_energy", round(ec,4), "semantic4_visual_energy", round(ev,6),
          "contrast_top4", np.round(sc[:4].numpy(),4).tolist())

expert_root = "paligemma_with_expert.gemma_expert.model.layers"
o12_name = f"{expert_root}.12.self_attn.o_proj.weight"
O12 = load_original(o12_name)
Q12 = subspaces[12]
O12edit = O12.clone()
for h in [2, 6]:
    sl = slice(256*h, 256*(h+1))
    Oh = O12[:, sl]
    O12edit[:, sl] = Oh + 0.5 * (Oh @ Q12) @ Q12.T
save_edit(o12_name, O12edit)
torch.save({"Q12": Q12, "heads": [2,6], "gain": 1.5,
            "edit": {o12_name: O12edit}}, W/"semantic_subspace_L12_gain150.pt")
print("Installed:", edited())
print("Relative output-matrix change:", round(float((O12edit-O12).norm()/O12.norm()),5))
with notes.open("a") as f:
    f.write(
        "\n## Pass20: matched initial-state contrast probe\n"
        "32 exact replay inputs, 16 task/semantic groups, VLM residual means before "
        "cache layers4/6/9/12/15. Candidate H: at L12, map normalized saved residual "
        "directions through existing RMSNorm/value weight geometry; remove top4 visual "
        "summary directions from paired language contrasts, take top4 contrast directions. "
        "Only expert L12H2/H6 output matrices changed: O_h <- O_h +0.5*(O_h Q)Q^T. "
        "No fitted targets or losses; this is a rank4 semantic-direction gain, not whole-head scaling. "
        "Mean summaries and nonidentical images limit the causal interpretation until rollout testing.\n"
    )
```
