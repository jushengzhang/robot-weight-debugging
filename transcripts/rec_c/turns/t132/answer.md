Head 1 keeps a stable empty-closure signal across denoising. Head 7's centered signal rises substantially on normal controls, which helps explain the unwanted lift. I will remove the head-7 route and center head 1 using all ten calls from the same source controls, without using reserved-check outcomes or success labels.
```python
h = 1
beta_all = -float(S41[ctrl,:,h].mean()/S42[ctrl,:,h].mean())
feature_h1 = cd["feature"] + beta_all*cd["reference"]
delta_h1 = torch.outer(parts[h], feature_h1)
for name in list(edited()):
    drop_edit(name)
save_edit(vname, load_original(vname) + delta_h1)
canddir = W/"candidate_H1_centered_clearance_060"
canddir.mkdir(exist_ok=True)
np.save(canddir/(vname+".npy"), load(vname).numpy())
h1design = {
    "feature": cd["feature"], "reference":cd["reference"], "beta":beta_all,
    "value_direction":parts[h], "delta":delta_h1, "nominal":0.6,
    "source_control_indices":ctrl,
    "centering":"source control attention scores averaged over all ten calls",
}
torch.save(h1design, W/"H1_centered_clearance_design.pt")
print("head1 beta: initial-call",cd["betas"][1],"all-call",beta_all)
for group in dict.fromkeys(group_key(r) for r in m):
    inds = [i for i,r in enumerate(m) if group_key(r)==group]
    scores = S41[inds,:,1] + beta_all*S42[inds,:,1]
    print(group,"all-call score mean",round(float(scores.mean()),3),
          "scene SD",round(float(scores.mean(1).std()),3))
print("rank",1,"relative weight change",round(float(delta_h1.norm()/load_original(vname).norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## Temporal centering diagnosis after pass43\n"
            "H7 centered source-contact score grows0.028->2.944 across denoising calls; "
            "open-control grows-0.024->2.313. H1 stays selective: empty~3/source and~2.2/check, "
            "contact~-0.4, open~0.1. Reconstruction residual grows to10.7% at final call, "
            "so later recovered scores are approximate.\n"
            "New candidate removes H7 routing entirely. Rank1 VLM17 v_proj, H1 route at nominal0.60; "
            f"normal-source all-call centering coefficient={beta_all:.6f}. "
            "Same14 contact+16 open source controls, no success labels or check outcomes used. "
            "Fresh confirmation_batch3 still untouched.\n")
```
