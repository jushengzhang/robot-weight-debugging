I’ll assemble the three masked-video readout slices and remove all 188 diagnostic edits. Then I’ll measure how much raw-versus-clean difference remains after detected suffix tokens are excluded; that isolates contamination of the instruction context from direct attention to the suffix.
```python
torch.set_num_threads(12)
exec((W/"masked_readout.py").read_text())
z2 = collect_masked_readout(28)
clear_edits()
spec = torch.load(W/"masked_readout_spec.pt", weights_only=False)
M = torch.zeros(128, len(spec["queries"]), 3072, dtype=torch.float64)
for j in range(3):
    z = np.load(W/f"masked_video_{j}.npy")
    ids = spec["feature_ids"][j*spec["m"]:(j+1)*spec["m"]]
    M[:,:,ids] = torch.from_numpy(z)
torch.save(M.float(), W/"masked_video_features.pt")
U = torch.load(W/"weighted_video_features.pt", weights_only=False).double()
print("masked", tuple(M.shape), "unmasked", tuple(U.shape), "edits", len(edited()))
for qi, q in enumerate(spec["queries"][:13]):
    dm = M[:48,qi] - M[48:96,qi]
    du = U[:48,qi] - U[48:96,qi]
    print(qi, q, "contrast_rms", round(dm.square().mean().sqrt().item(),6),
          "unmasked_rms", round(du.square().mean().sqrt().item(),6),
          "clean_readout_difference", round((M[48:,qi]-U[48:,qi]).square().mean().sqrt().item(),6))
print("\nPRIOR FINAL NOTES")
print((W/"prior_notes.md").read_text()[16500:23000])
print("\nREPLAY PATHS")
rp=json.loads((W/"replay_paths.json").read_text())
print(str(rp)[:1200])
with (W/"notes.md").open("a") as f:
    f.write("\nPass28 masked video readouts assembled into masked_video_features.pt; all188 instrumentation edits cleared. Post-restart analysis begins with no active edits.\n")
```
