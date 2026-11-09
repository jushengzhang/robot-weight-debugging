I will construct the reopening route algebraically from the saved physical-readout constraints. It will use the same visual feature, target only head 1, and aim to change gripper opening plus a smaller vertical clearance while leaving the other first-order physical readouts unchanged. The next probe must verify those effects.
```python
target_open = torch.zeros(8,7,dtype=torch.float64)
target_open[1,2] = -0.15*scale
target_open[1,6] = 2.0*scale
u_open = (torch.linalg.pinv(C) @ target_open.flatten()).float()
delta_open = torch.outer(u_open, feature_h1)
print("constraint residual",float((C@u_open.double()-target_open.flatten()).norm()))
print("value direction norm",float(u_open.norm()))
print("relative weight change",float(delta_open.norm()/load_original(vname).norm()))
print("Original contact positive-gripper fraction",
      float(np.mean(a44["orig"][[i for i,r in enumerate(m) if group_key(r)=="source_contact"],:5,6]>0)))
save_edit(vname, load_original(vname) + delta_open)
reset_design = {
    "feature":feature_h1, "value_direction":u_open, "delta":delta_open,
    "target":target_open.float(), "nominal_clearance":0.15,
    "nominal_opening":2.0, "centering_beta":beta_all,
    "constraints":"H1 Z and gripper only; other heads/physical readouts zero first-order",
}
torch.save(reset_design,W/"empty_grasp_reset_design.pt")
p = W/"candidate_empty_grasp_reset"
p.mkdir(exist_ok=True)
np.save(p/(vname+".npy"), load(vname).numpy())
with (W/"notes.md").open("a") as f:
    f.write("\n## Final alternative: empty-grasp reset, not lift alone\n"
            "Same H1 centered empty-closure feature; direct geometric value route targets nominal "
            "opening2.0 and clearance0.15, with all other physical/head first-order constraints zero. "
            "No success-label fitting. Needs fixed-input causal check, paired rollout, independent validation. "
            "Positive executed gripper is closing (contact controls predominantly positive).\n")
```
