The early texture contrasts are low-rank, but they overlap task variation, so removing the entire contrast subspace could be harmful. I will first check head-level texture sensitivity and use low-perturbation light observations as a reference for preserving normal attention outputs.
```python
def token_difference(a, b):
    xa = torch.load(a, map_location="cpu", weights_only=False)["input_image"].float()
    xb = torch.load(b, map_location="cpu", weights_only=False)["input_image"].float()
    d = (xa-xb).abs().mean(1, keepdim=True)
    return torch.nn.functional.adaptive_avg_pool2d(d, (7,14)).flatten().numpy()

attnstats = {}
for L in [0,1,2,5]:
    data = []
    for j in [0,2]:
        a,b = p4[j],p4[j+1]
        M = np.load(a["attention"][str(L)])
        N = np.load(b["attention"][str(L)])
        change = token_difference(a["input"], b["input"])
        pm = M.mean(1); pn=N.mean(1)
        mass = 0.5*(pm.sum(-1)+pn.sum(-1))
        # Normalize within video for spatial comparisons; retain video mass separately.
        pm=pm/pm.sum(-1,keepdims=True).clip(1e-12)
        pn=pn/pn.sum(-1,keepdims=True).clip(1e-12)
        roi = change > np.quantile(change, 0.60)
        enrich = ((pm+pn)/2)[:,roi].sum(-1)/roi.mean()
        tv = np.abs(pm-pn).sum(-1)/2
        ent = -(0.5*(pm+pn)*np.log(0.5*(pm+pn)+1e-12)).sum(-1)/np.log(98)
        data.append(np.stack([mass,enrich,tv,ent],1))
    H=np.mean(data,axis=0)
    attnstats[L]=H
    order=np.argsort(-(H[:,1]*H[:,2]))
    print("\nL",L,"head, video mass, changed-region enrichment, spatial TV, entropy, contrast overlap")
    for h in order[:8]:
        print(h, *np.round(H[h],3), round(float(headscores[L][h]),4))
with np.load(r2[0]["activations"]) as z:
    print("\nRollout activation key examples:", z.files[:4], [k for k in z.files if "video" in k][:5])
    print("Total keys:",len(z.files))
# 37 matched low-difficulty light initial observations, never averaging over diverged trajectories.
rlookup = {(r["suite"],r["id"]):r for r in r2}
lightpaths = []
for r in r2:
    if r["category"] == "Light Conditions":
        lightpaths.append(str(Path(r["frames_dir"])/"t0000_inputs.pt"))
(W/"light_initial_paths.json").write_text(json.dumps(lightpaths))
all_initial_paths = paths0 + lightpaths
(W/"diagnostic_initial_all_paths.json").write_text(json.dumps(all_initial_paths))
print("Probe reference inputs:",len(all_initial_paths))
print("Light path examples:",lightpaths[:2])
np.savez(W/"early_attention_map_stats.npz", **{f"L{L}":v for L,v in attnstats.items()})
```