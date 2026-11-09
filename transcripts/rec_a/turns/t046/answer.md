I will check whether the matched images yield a consistent gripper-state direction across scenes. If they do, I can test attenuating that direction where the expert reads visual values, rather than disabling whole heads or changing the action readout.
```python
p15 = probe_records(15)
p15 = sorted(p15, key=lambda r:r["idx"])
a15 = [np.load(local_path(r["activations"])) for r in p15]
ac15 = np.stack([np.load(local_path(r["action"])) for r in p15])
meta15 = []
for r in p15:
    v = torch.load(local_path(r["input"]), map_location="cpu", weights_only=False)
    meta15.append(v)
print("Visual contrasts, closed minus open; n=24 pairs")
for im in ("observation/image","observation/wrist_image"):
    dif = np.stack([meta15[2*j+1]["element"][im].astype(float)-
                    meta15[2*j]["element"][im].astype(float) for j in range(24)])
    patch = np.mean(np.abs(dif),axis=(0,3)).reshape(16,14,16,14).mean((1,3))
    inds = np.unravel_index(np.argsort(patch.ravel())[-8:][::-1],patch.shape)
    print(im, "MAE", round(float(np.abs(dif).mean()),2),
          "largest changed patches:", list(zip(inds[0].tolist(),inds[1].tolist())))
states = np.stack([x["element"]["observation/state"] for x in meta15])
print("State differences (closed-open), mean abs:",np.round(np.abs(states[1::2]-states[::2]).mean(0),4).tolist())
print("Gripper outputs open/closed means:",
      np.round(ac15[::2,:,6].mean(1),2).tolist(), np.round(ac15[1::2,:,6].mean(1),2).tolist())

visdirs = {}
print("Layer: wrist contrast consistency, base/wrist contrast-norm ratio, normalized value-direction separation")
for l in range(17):
    X = np.stack([x[f"mean::vlm.{l}.resid::wrist"] for x in a15])
    B = np.stack([x[f"mean::vlm.{l}.resid::base"] for x in a15])
    dx = X[1::2] - X[::2]
    db = B[1::2] - B[::2]
    d = dx.mean(0)
    cs = dx @ d / (np.linalg.norm(dx,axis=1)*np.linalg.norm(d)+1e-12)
    layer = l+1
    # Gemma RMSNorm's learned scale is 1+weight; this is weight analysis, not a forward pass.
    normname = V + f"layers.{layer}.input_layernorm.weight"
    scale = 1 + load_original(normname)
    dn = torch.tensor(d) * scale
    wv = load_original(V+f"layers.{layer}.self_attn.v_proj.weight")
    vd = wv @ dn
    vd = vd / vd.norm()
    visdirs[layer] = {"residual_contrast": torch.tensor(d), "normalized_input_direction":dn/dn.norm(),
                      "value_direction":vd, "cosines":torch.tensor(cs)}
    print(layer, round(float(np.mean(cs)),3), round(float(np.min(cs)),3),
          round(float(np.linalg.norm(db.mean(0))/np.linalg.norm(d)),3),
          round(float(vd.norm()),2))
torch.save(visdirs, W / "visual_gripper_directions.pt")

# Separate a consistently changed sensory feature from an outcome-selected contrast.
print("Value-direction similarity between first and second half of scene pairs:")
for layer in (1,3,6,9,12,15,16,17):
    l=layer-1
    X=np.stack([x[f"mean::vlm.{l}.resid::wrist"] for x in a15])
    dx=torch.from_numpy(X[1::2]-X[::2])
    scale=1+load_original(V+f"layers.{layer}.input_layernorm.weight")
    wv=load_original(V+f"layers.{layer}.self_attn.v_proj.weight")
    d1=wv@(dx[:12].mean(0)*scale); d2=wv@(dx[12:].mean(0)*scale)
    print(layer, round(float(torch.nn.functional.cosine_similarity(d1,d2,dim=0)),3))

with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 15: visual-feedback localization\n"
            "- Counterfactual fixed-noise copies were refused (no pass charged); only original rollout inputs used.\n"
            "- Probed 24 matched t0 open/closed pairs; VLM residuals are independent of the differing action noise.\n"
            "- Computing a matched visual gripper-state contrast, not a success/failure classifier or fitted weight model.\n")
```
