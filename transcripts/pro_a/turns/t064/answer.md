The zero-change control reproduced every success/failure outcome, so the paired success comparison is sound; the small variation in steps and gripper counts means those secondary metrics should be treated cautiously. I will now test whether the instruction bottleneck is in selecting keys rather than transmitting values: amplify only the instruction-contrast subspace of layer 12’s cached keys, leaving the dominant visual directions untouched.
```python
r23 = read_jsonl(W/"roll_23.jsonl")
arms23 = defaultdict(dict)
for rr in r23:
    arms23[rr["id"]][rr["model"]] = rr
dmax = []
for arms in arms23.values():
    a = np.load(arms["original"]["actions_npy"])
    b = np.load(arms["edited"]["actions_npy"])
    n = min(len(a),len(b))
    dmax.append(float(np.max(np.abs(a[:min(n,5)]-b[:min(n,5)]))))
print("No-op first5 action max differences: median/max", float(np.median(dmax)), max(dmax))
with notes.open("a") as f:
    f.write(
        "\n## Pass23: no-op control\n"
        "Saved position tensor exactly equals original. Dev69->69/128, 0 discordant outcomes. "
        "Secondary metrics not bitwise stable: mean success steps154.6 vs154.7 and loops30 vs28. "
        "Interpret success-pair comparisons primarily; repeated semantic rescues are not unique "
        "mechanistic evidence for individual edits.\n"
    )
for n in list(edited()):
    drop_edit(n)

l = 12
gamma = 1 + load_original(f"{vlm_root}.{l}.input_layernorm.weight")
k12_name = f"{vlm_root}.{l}.self_attn.k_proj.weight"
K12 = load_original(k12_name)
XV = torch.from_numpy(np.stack([z[f"mean::vlm.{l-1}.resid::{m}"]
                               for z in z20 for m in ["base", "wrist"]])).float()
XL = torch.from_numpy(np.stack([z[f"mean::vlm.{l-1}.resid::lang"] for z in z20])).float()
XV = torch.nn.functional.normalize(XV, dim=1)
XL = torch.nn.functional.normalize(XL, dim=1)
AK = K12 * gamma[None,:]
YV, YL = XV @ AK.T, XL @ AK.T
_, svk, vhvk = torch.linalg.svd(YV, full_matrices=False)
Qvk = vhvk[:4].T
Ck = torch.stack([YL[g["task"]] - YL[g["semantic"]] for _,g in sorted(groups.items())])
Ckc = Ck - (Ck @ Qvk) @ Qvk.T
_, sck, vhck = torch.linalg.svd(Ckc, full_matrices=False)
Qk = vhck[:4].T.contiguous()
K12edit = K12 + (Qk @ (Qk.T @ K12))  # 2x only on this rank-4 output subspace
save_edit(k12_name, K12edit)
torch.save({"Qk":Qk, "edit":{k12_name:K12edit}}, W/"semantic_key_L12_gain200.pt")
print("Key-subspace energies: visual, instruction contrast",
      round(float((YV@Qk).square().sum()/YV.square().sum()),5),
      round(float((Ck@Qk).square().sum()/Ck.square().sum()),4))
print("Relative key-matrix change:", round(float((K12edit-K12).norm()/K12.norm()),4))
print("Expert residual contrast/pooled norm by layer:")
for l in [4,6,9,12,15]:
    X = torch.from_numpy(np.stack([z[f"mean::expert.{l}.resid::action"] for z in z20])).float()
    D = torch.stack([X[g["task"]]-X[g["semantic"]] for g in groups.values()])
    print(l, round(float(D.norm(dim=1).median()/X.norm(dim=1).median()),4))
with notes.open("a") as f:
    f.write(
        "\nCandidate J: same matched contrast geometry, now key space at VLM L12. "
        "Remove top4 pooled visual key directions from language contrasts; take top4 residual "
        "directions Qk. K <- K + Qk Qk^T K (2x in rank4 subspace only). "
        "Tests semantic selection rather than head/value gain; all other weights original.\n"
    )
```
