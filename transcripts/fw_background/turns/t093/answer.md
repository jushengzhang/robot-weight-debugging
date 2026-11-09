No direction passed that strict split-half screen, so I won’t edit the readout on that basis. I’ll check one remaining distinction in the existing data: the matched background-versus-light **mean shift**, rather than variation between two different textures. A consistent shift could reveal a bias that the high-versus-low texture comparison misses.
```python
import re
def variant_key(path):
    m = re.search(r"(libero_(?:10|goal|object|spatial))_v(\d+)_s", str(path))
    assert m, path
    return m.group(1), int(m.group(2))

manifest19 = json.loads((W / "fixed_residual_probe_manifest.json").read_text())
source_keys = [variant_key(p) for p in manifest19["source_paths"]]
ix_by_key = {k:i for i,k in enumerate(source_keys)}
pairs_meta = json.loads((W / "diagnostic_pairs.json").read_text())
triples = []
for r in pairs_meta:
    ks = [(r["suite"],r.get(t)) for t in ["high","low","light"]]
    if all(k in ix_by_key for k in ks):
        triples.append([ix_by_key[k] for k in ks])
print("Matched high/low/light task triples:",len(triples))
T = np.array(triples,dtype=int)
assert len(T) >= 20

mean_shift_stats, mean_shift_data = {}, {}
for L in range(30):
    XR = torch.from_numpy(get_resid("action",L)[0][:,0]).float()
    C = XR[54:]
    m = C.mean(0)
    CL = C-m
    _,ss,vv = torch.linalg.svd(CL,full_matrices=False)
    frac = torch.cumsum(ss.square(),0)/ss.square().sum()
    r = int(torch.searchsorted(frac,torch.tensor(.95)))+1
    q = torch.linalg.qr(torch.cat([m[:,None],torch.ones(1024,1),vv[:r].T],1),mode="reduced").Q
    delta = (XR[T[:,0]]+XR[T[:,1]])/2-XR[T[:,2]]
    dp = delta-(delta@q)@q.T
    raw = dp.mean(0)
    u = raw/raw.norm().clamp_min(1e-12)
    c0,c1 = dp[::2].mean(0),dp[1::2].mean(0)
    stab = float(torch.nn.functional.cosine_similarity(c0,c1,dim=0))
    mean_shift_stats[L] = {
        "split_mean_cosine":stab,"protect_rank":r,
        "mean_texture_projection":float(dp@u @ torch.ones(len(dp))/len(dp)),
        "light_projection_rms":float((CL@u).square().mean().sqrt()),
        "texture_projection_rms":float((delta@u).square().mean().sqrt()),
        "contrast_fraction":float((delta@u).square().sum()/delta.square().sum()),
        "signed_consistency":float(((dp@u)>0).float().mean()),
        "head_mapping_norm":float((H@u).norm()) if L==29 else None
    }
    mean_shift_data[f"u{L}"] = u.numpy()
    mean_shift_data[f"Q{L}"] = q.numpy()
    mean_shift_data[f"mean{L}"] = m.numpy()

print("L split_cos rank mean_shift light_rms texture_rms contrast_frac sign_fraction")
for L,r in mean_shift_stats.items():
    print(L,*[round(float(v),4) for v in [
        r["split_mean_cosine"],r["protect_rank"],r["mean_texture_projection"],
        r["light_projection_rms"],r["texture_projection_rms"],
        r["contrast_fraction"],r["signed_consistency"]]])
np.savez_compressed(W / "matched_mean_shift_subspaces.npz",**mean_shift_data)
(W / "matched_mean_shift_stats.json").write_text(json.dumps(mean_shift_stats,indent=2))
print("No edits:", edited())
```
