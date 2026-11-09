The DC-subspace direction is reproducible across tasks (split cosine 0.969) and has about 53 times more contrast energy than light-task variation. I’ll attenuate only that rank-one component of the patch embedding’s DC mode by 25%. The horizontal, vertical, and checkerboard modes, patch bias, and all transformer weights will remain unchanged. A fixed-input probe will test the limitation of using episode-averaged video features to identify it.
```python
# Prepare a low-texture, protected DC direction as a norm-matched control.
Qall = torch.linalg.qr(torch.cat([Qdc,udc[:,None]],1),mode="complete").Q
Z = Qall[:,Qdc.shape[1]+1:]
_,_,vd_control = torch.linalg.svd(Ddc@Z,full_matrices=False)
c_dc = Z@vd_control[-1]
c_out = Bdc@c_dc
target_norm = float((d_out@DC).norm())
control_norm = float((c_out@DC).norm())
control_scale = target_norm/control_norm

np.savez_compressed(W/"patch_dc_causal_directions.npz",
                    target=d_out.numpy(),control=c_out.numpy(),
                    control_scale=np.array(control_scale,dtype=np.float64))

patch_utility = '''
def set_patch_dc_edit(dose, kind="target"):
    W = Path(WORK)
    for name in list(edited()):
        drop_edit(name)
    z = np.load(W/"patch_dc_causal_directions.npz")
    direction = torch.from_numpy(z[kind].copy())
    actual_dose = dose*(float(z["control_scale"]) if kind=="control" else 1.0)
    name = "mot.mixtures.video.patch_embedding.weight"
    original = load_original(name)
    flat = original.reshape(3072,48,4)
    dc = flat.sum(-1)/2
    correction = torch.outer(direction,direction@dc)[:,:,None]/2
    modified = (flat-actual_dose*correction).reshape_as(original)
    if dose:
        save_edit(name,modified)
    spec = {"family":"protected_patch_DC_texture_direction",
            "dose":dose,"actual_dose":actual_dose,"kind":kind,
            "tensor":name,"rank":1,
            "relative_change":float((modified-original).norm()/original.norm())}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
'''
with (W/"experiment_utils.py").open("a") as f:
    f.write(patch_utility)
exec(patch_utility)
spec = set_patch_dc_edit(.25)
P1 = load("mot.mixtures.video.patch_embedding.weight")
mode_delta = ((P1-P0).reshape(3072,48,4)@haar.T).square().sum((0,1)).sqrt()
print("Applied:",spec)
print("Changed mode norms [DC,H,V,HH]:",mode_delta.tolist())
print("Control dose multiplier:",control_scale)
print("Control texture/light ratio:",
      float((Ddc@c_dc).square().mean()/(Cc@c_dc).square().mean().clamp_min(1e-12)))

snapshot = W/"candidates"/"patch_dc_d025"
snapshot.mkdir(exist_ok=True)
np.save(snapshot/"patch.weight.npy",P1.numpy())
(snapshot/"candidate.json").write_text(json.dumps(spec,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass26 follow-up: input/DC localization\n")
    f.write("- Failure-specific MLP directions were not shared across tasks in late layers; no MLP edit made.\n")
    f.write("- Pixel-mode analysis refuted high-frequency texture filtering: BG/light variance ratio ~3-3.4 for DC, only~1-1.8 for other modes. No checkerboard filter applied.\n")
    f.write("- Early video residual contrasts projected into the 48-dimensional patch DC output subspace, protecting both camera light means and top8 light PCs (96.1% variance). Direction split cosine .9690, contrast/light ratio53.38, light variance fraction .00295.\n")
    f.write("- New experimental edit: patch DC rank1 projection, dose .25; all other spatial modes and all transformer weights unchanged. Episode-average source confounding must be checked by fixed-input probe. Dose .25/.5 and matched low-texture DC control prepared.\n")
```
