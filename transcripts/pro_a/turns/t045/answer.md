The isolated row-193 edit did not generalize: validation fell by 2.3 points, with losses on object and position perturbations. I will reject it. Next I will examine the much larger common visual direction in the projector and test a stronger, rank-one attenuation; unlike a global image-scale change, this leaves the other visual directions unchanged.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass16: independent validation of row193 keep20%\n"
        "82->79/128, -2.3 +/-2.6pp, 4 wins/7 losses. "
        "Environment13->15, object24->21, position11->9, semantic22->21, task12->13. "
        "Old-goal failures6 unchanged. REJECT row193-only candidate: development gain did not generalize.\n"
    )
for n in list(edited()):
    drop_edit(n)

proj_name = next(n for n in NAMES if n.endswith("multi_modal_projector.linear.weight"))
M = load_original(proj_name)
U, S, Vh = torch.linalg.svd(M, full_matrices=False)
print("Projector top6 singular values:", np.round(S[:6].numpy(), 3).tolist())
u0, v0, s0 = U[:, 0], Vh[0], S[0]
M50 = M - 0.5 * s0 * torch.outer(u0, v0)
save_edit(proj_name, M50)
torch.save({proj_name: M50}, W / "projector_keep050.pt")
print("Installed rank-one edit; relative Frobenius change:", round(float((M50-M).norm()/M.norm()), 4))

# Inspect the existing per-input residual summaries, without running the network.
pr3 = json.loads((W / "probe_3" / "results.json").read_text())
Z = [np.load(r["activations"]) for r in pr3 if "activations" in r]
for l in [0, 3, 6, 9, 12, 15, 17]:
    bk, lk = f"mean::vlm.{l}.resid::base", f"mean::vlm.{l}.resid::lang"
    b = np.stack([z[bk] for z in Z if bk in z])
    t = np.stack([z[lk] for z in Z if lk in z])
    mu_b, mu_t = b.mean(0), t.mean(0)
    def coss(a, b):
        return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    print("VLM", l, "N", len(b),
          "base/lang mean-norm", round(float(np.linalg.norm(mu_b)), 1), round(float(np.linalg.norm(mu_t)), 1),
          "cos", round(coss(mu_b, mu_t), 3),
          "base variation/mean", round(float(np.sqrt(((b-mu_b)**2).sum(1).mean())/np.linalg.norm(mu_b)), 3),
          "cos_projector_top", round(coss(mu_b, u0.numpy()), 3))
print("Roll15 auxiliary files:", [str(p.relative_to(W)) for p in (W/"fwd"/"roll_15").glob("*.json")])
print("Dev episode IDs:", [r["id"] for r in r1])
with notes.open("a") as f:
    f.write(
        "\nCandidate G: reset position edits; projector top singular mode retain50% "
        "(earlier retain80% gave only semantic gain). All other singular modes unchanged. "
        "Test dose-response of common visual carrier, not overall visual scaling.\n"
    )
```
