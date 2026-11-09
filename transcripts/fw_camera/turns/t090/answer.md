There are 128 camera variants that have never been run; I’ll reserve them for a final check. For diagnosis, I’ll use only the existing matched-state data to test whether the two-direction readout edit omitted most of the camera-sensitive signal, and whether a broader subspace is reproducible across tasks.
```python
print("Probe10 files (non-npy):", [p.name for p in (root / "probe_10").iterdir() if p.suffix not in (".npy", ".npz")])
print("Probe12 request prefix:", (root / "request_12.json").read_text()[:800])
print("Diagnostic input structure:", str(json.loads((root / "diagnostic_input_paths.json").read_text()))[:700])
print("Replay input structure:", str(json.loads((root / "compact_replay_inputs.json").read_text()))[:700])
bz = np.load(root / "readout_protected_camera_subspace_broad.npz")
P = torch.from_numpy(bz["protect_basis"]).double()
Vcam = torch.from_numpy(bz["camera_basis"]).double().T
s = torch.from_numpy(bz["camera_singular"]).double()
W0 = load_original(wname).double()
print("Readout protected fraction:", float((W0 @ P).square().sum()/W0.square().sum()))
print("Broad camera spectrum, cumulative energy and W-edit relative norms:")
for r in (1,2,4,8,12,16,24,32,40,56):
    U = Vcam[:, :r]
    print(r, "energy", round(float(s[:r].square().sum()/s.square().sum()),4),
          "full_edit_rel", round(float(torch.linalg.norm(W0@U)/torch.linalg.norm(W0)),4),
          "row_rel", np.round((torch.linalg.vector_norm(W0@U, dim=1)/
                               torch.linalg.vector_norm(W0, dim=1)).numpy(),3).tolist())
print("Protection rank:", P.shape[1], "clean rank:", int(bz["clean_rank"]))
unused_camera = [v for v in practice if v["category"]=="Camera Viewpoints" and (v["suite"],v["id"]) not in seen_keys]
unused_camera.sort(key=lambda v: hashlib.sha256(f'final-camera-reserve:{v["suite"]}:{v["id"]}'.encode()).hexdigest())
final_camera_reserve = [{"suite": v["suite"], "id": v["id"], "seed": 160000+i} for i,v in enumerate(unused_camera)]
(root / "final_camera_reserve.json").write_text(json.dumps(final_camera_reserve, indent=2))
print("Final reserve saved:", len(final_camera_reserve), "variants, not yet run")
```
