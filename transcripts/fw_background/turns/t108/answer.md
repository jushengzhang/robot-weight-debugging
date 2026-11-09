I’ll extract failure-specific MLP directions while protecting variation from both light controls and successful backgrounds. Split checks will be by task, not by variant, so two failures of the same task cannot make a direction appear more reproducible than it is.
```python
p26 = probe_results(26)
mlp_acts = [dict(np.load(r["activations"])) for r in p26]
print("MLP key examples:",list(mlp_acts[0])[:3])
print("MLP output projection names:")
for n in NAMES:
    if "mot.mixtures.action.blocks.20." in n and (".ffn." in n or ".mlp." in n or "modulation" in n):
        print(n,SHAPES[n])

task_by_variant = {}
for r in pairs_meta:
    for field in ["high","low","light"]:
        if r.get(field) is not None:
            task_by_variant[(r["suite"],r[field])] = (r["suite"],r["task"])
F = np.array(failure_pairs,dtype=int)
task_groups = defaultdict(list)
for j,(bg,ref) in enumerate(F):
    task_groups[task_by_variant[source_keys[bg]]].append(j)
group_indices = list(task_groups.values())
print("Independent failure tasks:",len(group_indices),
      "group sizes:",[len(x) for x in group_indices])

control_ix = np.array(success_bg_indices+list(range(54,91)),dtype=int)
mlp_data,mlp_stats = {},{}
for L in range(30):
    key = f"mean::action.{L}.mlp_out::action"
    X = torch.from_numpy(np.stack([r[key] for r in mlp_acts])).float()
    C = X[control_ix]
    means = torch.stack([X[success_bg_indices].mean(0), X[54:].mean(0)])
    centered = torch.cat([X[success_bg_indices]-means[0],X[54:]-means[1]],0)
    _,ss,vv = torch.linalg.svd(centered,full_matrices=False)
    energy = torch.cumsum(ss.square(),0)/ss.square().sum()
    rank = int(torch.searchsorted(energy,torch.tensor(.95)))+1
    q = torch.linalg.qr(torch.cat([means.T,torch.ones(1024,1),vv[:rank].T],1),mode="reduced").Q
    D = X[F[:,0]]-X[F[:,1]]
    DP = D-(D@q)@q.T
    taskD = torch.stack([DP[ix].mean(0) for ix in group_indices])
    direction = taskD.mean(0)
    direction = direction/direction.norm().clamp_min(1e-12)
    h0,h1 = taskD[::2].mean(0),taskD[1::2].mean(0)
    stab = float(torch.nn.functional.cosine_similarity(h0,h1,dim=0))
    ctrl_var = float((centered@direction).square().mean())
    err_var = float((D@direction).square().mean())
    proj = (taskD@direction)
    mlp_stats[L] = {
        "protect_rank":rank,"split_task_mean_cosine":stab,
        "failure_to_control_ratio":err_var/max(ctrl_var,1e-12),
        "failure_fraction":float((D@direction).square().sum()/D.square().sum()),
        "control_fraction":float((centered@direction).square().sum()/centered.square().sum()),
        "failure_projection_rms":math.sqrt(err_var),
        "control_projection_rms":math.sqrt(ctrl_var),
        "signed_task_consistency":float((proj>0).float().mean())
    }
    mlp_data[f"u{L}"] = direction.numpy()
    mlp_data[f"Q{L}"] = q.numpy()
    mlp_data[f"mean{L}"] = C.mean(0).numpy()
    mlp_data[f"D{L}"] = D.numpy()
    mlp_data[f"C{L}"] = C.numpy()

print("\nL protect_rank split_cos ratio fail_frac control_frac fail_RMS control_RMS")
for L,s in mlp_stats.items():
    print(L,*[round(float(v),4) for v in [
        s["protect_rank"],s["split_task_mean_cosine"],s["failure_to_control_ratio"],
        s["failure_fraction"],s["control_fraction"],s["failure_projection_rms"],
        s["control_projection_rms"]]])
np.savez_compressed(W/"failure_mlp_subspaces.npz",**mlp_data)
(W/"failure_mlp_stats.json").write_text(json.dumps(mlp_stats,indent=2))
```
