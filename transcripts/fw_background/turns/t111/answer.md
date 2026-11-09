I’ll compare the four orthogonal 2×2 spatial modes—mean, horizontal, vertical, and checkerboard—at several image scales. This is only an input diagnostic, not a model forward pass. The useful signal would be excess texture-driven variation in the checkerboard mode relative to ordinary light/task variation, especially in the reliably failing cases.
```python
images = torch.cat([
    torch.load(path,map_location="cpu",weights_only=False)["input_image"].float()
    for path in manifest19["source_paths"]
],dim=0)

haar = torch.tensor([[1,1,1,1],[1,-1,1,-1],
                     [1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float32)/2
freq_stats = {}
for scale in [1,2,4,8,16]:
    n,c,h,w = images.shape
    Y = images.reshape(n,c,h//scale,scale,w//scale,scale).mean((3,5))
    V = torch.stack([Y[:,:,0::2,0::2],Y[:,:,0::2,1::2],
                     Y[:,:,1::2,0::2],Y[:,:,1::2,1::2]],-1)
    M = V@haar.T
    paired_delta = M[:54:2]-M[1:54:2]
    light_delta = M[54:]-M[54:].mean(0,keepdim=True)
    failed_delta = M[F[:,0]]-M[F[:,1]]
    e_pair = paired_delta.square().mean((0,1,2,3))
    e_light = light_delta.square().mean((0,1,2,3))
    e_failed = failed_delta.square().mean((0,1,2,3))
    freq_stats[scale] = {
        "background_mode_energy":e_pair.tolist(),
        "light_task_mode_energy":e_light.tolist(),
        "background_light_ratio":(e_pair/e_light.clamp_min(1e-12)).tolist(),
        "failure_light_ratio":(e_failed/e_light.clamp_min(1e-12)).tolist()
    }
    print("Scale",scale,"[DC,H,V,HH]")
    print(" texture/light:",np.round((e_pair/e_light).numpy(),3).tolist())
    print(" failure/light:",np.round((e_failed/e_light).numpy(),3).tolist())
(W/"input_spatial_frequency_diagnostics.json").write_text(json.dumps(freq_stats,indent=2))

P0 = load_original("mot.mixtures.video.patch_embedding.weight")
patch_modes = P0.reshape(3072,48,4)@haar.T
print("Patch weight mode energies:",patch_modes.square().sum((0,1)).tolist())
print("Current edits:",edited())
```
