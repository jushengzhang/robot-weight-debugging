A single-head ablation may miss gripper-state information carried through several heads. I will measure the visual difference between matched open- and closed-gripper starts, then prepare a fixed-noise probe to separate that visual feedback from the noise difference found earlier. This is a diagnostic control, not an inference-time change.
```python
E = "paligemma_with_expert.gemma_expert.model."
V = "paligemma_with_expert.paligemma.model.language_model."
outw = load_original("action_out_proj.weight")
g = outw[6] / outw[6].norm()

print("Action readout row norms:", np.round(outw.norm(dim=1).numpy(), 3).tolist())
print("Grip/pose readout cosines:", np.round((outw[:6] @ g / outw[:6].norm(dim=1)).numpy(), 3).tolist())
print("Late MLP readout coupling: layer, Frobenius norm, grip energy fraction, largest column share")
for l in range(10,18):
    dw = load_original(E + f"layers.{l}.mlp.down_proj.weight")
    coupling = g @ dw
    print(l, round(float(dw.norm()), 2),
          round(float(coupling.square().sum()/dw.square().sum()), 4),
          round(float(coupling.square().max()/coupling.square().sum()), 4))
    del dw

for k in (4,13):
    a = np.load(local_path(probe_records(k)[0]["activations"]))
    print("Probe", k, "activation types:", sorted(set(n.split("::")[1].rsplit(".",1)[-1] for n in a.files if "::" in n)))
print("p4 input labels:")
for x in p4:
    print(x["idx"], local_path(x["input"]).parent.name, local_path(x["input"]).name)

# Use 24 scene pairs, balanced by recovery level, all from the discovery rollout.
byid = {}
for r in r1:
    byid.setdefault(r["id"], {})[r["grip"]] = r
selected_ids = []
for lev in ("L1","L2","L3","L4"):
    ids = sorted(i for i,d in byid.items()
                 if len(d)==2 and d["open"]["level"]==lev
                 and not any(r["success_at_t0"] for r in d.values()))
    selected_ids.extend(ids[:6])
diagdir = W / "fixed_noise_visual_probe"
diagdir.mkdir(exist_ok=True)
fixed_noise = inp["noise"].copy()
fixed_manifest = []
for i in selected_ids:
    for grip in ("open","close"):
        r = byid[i][grip]
        src = local_path(r["frames_dir"]) / "t0000_inputs.pt"
        v = torch.load(src, map_location="cpu", weights_only=False)
        v["noise"] = fixed_noise.copy()
        dst = diagdir / f"s{i}_{grip}.pt"
        torch.save(v, dst)
        fixed_manifest.append({"id":i,"grip":grip,"input":str(dst),
                               "source":str(src),"success":r["success"],"level":r["level"]})
(W / "fixed_noise_visual_manifest.json").write_text(json.dumps(fixed_manifest, indent=2))
print("Fixed-noise diagnostic inputs:", len(fixed_manifest))
print("Input element fields:", {k:(v.shape if hasattr(v,"shape") else str(v)[:120]) for k,v in inp["element"].items()})
print("Paths for next probe:")
print(json.dumps([x["input"] for x in fixed_manifest]))
```
