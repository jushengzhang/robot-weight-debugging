The input data argues against high-frequency smoothing: texture changes are strongest in the local mean/DC mode, not the checkerboard mode. I won’t apply a high-pass filter. I’ll check whether the early video residuals contain a reproducible texture-sensitive direction within the patch embedding’s DC subspace; that would allow a rank-one edit while preserving all three edge-detail modes exactly.
```python
records14 = [json.loads(s) for s in (W/"roll_14.jsonl").read_text().splitlines() if s.strip()]
r14 = {(r["suite"],r["id"]):r for r in records14 if r["model"]=="original"}
sample_act = dict(np.load(next(r["activations"] for r in r14.values() if "activations" in r)))
video0keys = sorted(k for k in sample_act if k.startswith("mean::video.0.resid::"))
print("Video0 residual keys:",video0keys)
assert len(video0keys)==2

# Work in the 48-dimensional output subspace of the patch's spatial DC mode.
P0 = load_original("mot.mixtures.video.patch_embedding.weight")
PM = P0.reshape(3072,48,4)@haar.T
DC = PM[:,:,0]
Bdc = torch.linalg.qr(DC,mode="reduced").Q
video0 = {}
for key,r in r14.items():
    z = np.load(r["activations"])
    video0[key] = torch.from_numpy(np.stack([z[k] for k in video0keys])).float()

Ccam,Dcam,task_labels = [],[],[]
for r in pairs_meta:
    ck = (r["suite"],r.get("light"))
    if ck not in video0:
        continue
    Ccam.append(video0[ck]@Bdc)
    contrasts = []
    for field in ["high","low"]:
        bk = (r["suite"],r.get(field))
        if bk in video0:
            contrasts.append((video0[bk]-video0[ck])@Bdc)
    if contrasts:
        Dcam.append(torch.stack(contrasts).mean(0))
        task_labels.append((r["suite"],r["task"]))

Ccam = torch.stack(Ccam)
Dcam = torch.stack(Dcam)
means_dc = Ccam.mean(0)
Cc = (Ccam-means_dc).reshape(-1,48)
_,sc,vc = torch.linalg.svd(Cc,full_matrices=False)
Qdc = torch.linalg.qr(torch.cat([means_dc.T,vc[:8].T],1),mode="reduced").Q
Ddc = Dcam.reshape(-1,48)
Ddc_p = Ddc-(Ddc@Qdc)@Qdc.T
td = Ddc_p.reshape(len(Dcam),2,48).mean(1)
udc = td.mean(0)
udc = udc/udc.norm()
half_cos = float(torch.nn.functional.cosine_similarity(td[::2].mean(0),td[1::2].mean(0),dim=0))
d_out = Bdc@udc
dcstats = {
    "n_tasks":len(Dcam),"split_task_mean_cosine":half_cos,
    "contrast_light_ratio":float((Ddc@udc).square().mean()/(Cc@udc).square().mean().clamp_min(1e-12)),
    "contrast_fraction":float((Ddc@udc).square().sum()/Ddc.square().sum()),
    "light_variance_fraction":float((Cc@udc).square().sum()/Cc.square().sum()),
    "protected_light_variance":float(sc[:8].square().sum()/sc.square().sum()),
    "fraction_of_patch_weight_in_direction":float((d_out@DC).norm()/P0.norm()),
    "protected_mean_projection_max":float(torch.abs(means_dc@udc).max())
}
print("DC candidate geometry:",dcstats)
np.savez_compressed(W/"patch_dc_texture_direction.npz",
                    output_direction=d_out.numpy(),dc_basis=Bdc.numpy(),
                    feature_direction=udc.numpy(),protected=Qdc.numpy(),
                    dc_contrasts=Ddc.numpy(),control_centered=Cc.numpy())
(W/"patch_dc_texture_stats.json").write_text(json.dumps(dcstats,indent=2))
print("No edits:",edited())
```
