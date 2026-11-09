I will compare each head's attention with the image regions that actually changed, and measure how much its attention moves between the paired textures. I will also use the full 27-pair activation set to check whether the early shifts occupy a small, shared subspace suitable for a selective edit.
```python
torch.set_num_threads(min(12, os.cpu_count() or 1))
p4 = json.loads((W/"probe_4/results.json").read_text())
print("Attention metadata:", str(p4[0])[:2400])
inp = torch.load(paths0[0], map_location="cpu", weights_only=False)
print("Input keys/shapes:", {k:(tuple(v.shape) if hasattr(v,"shape") else str(v)[:220]) for k,v in inp.items()})
print("Probe_4 files:", [x.name for x in (W/"probe_4").iterdir()][:15])
with np.load(W/"probe_3/probe_orig_0_acts.npz") as z:
    print("Non-mean activation shapes:", [(k,z[k].shape) for k in z.files if not k.startswith("mean::")][:8])
# Contrast spectrum, task diversity, and overlap with each action output-head column space.
layerstats = {}
headscores = {}
basis = {}
for L in range(8):
    key = f"mean::action.{L}.attn_out::action"
    X = np.stack([a[key] for a in A3])
    D = X[::2]-X[1::2]
    C = 0.5*(X[::2]+X[1::2]); C -= C.mean(0)
    _, s, vt = np.linalg.svd(D, full_matrices=False)
    U = torch.from_numpy(vt[:4].T.copy())
    basis[L] = U
    ow = load_original(f"mot.mixtures.action.blocks.{L}.self_attn.o.weight")
    # Energy of each head's output columns inside texture-contrast subspace.
    htot = ow.reshape(1024,24,128).square().sum((0,2))
    hproj = (U.T@ow).reshape(4,24,128).square().sum((0,2))
    ratio = hproj / htot.clamp_min(1e-12)
    headscores[L] = ratio.numpy()
    layerstats[L] = {
        "contrast_rms": float(np.sqrt(np.mean(D**2))),
        "contrast_to_task": float(np.sqrt(np.mean(D**2)/np.mean(C**2))),
        "rank4": float(np.sum(s[:4]**2)/np.sum(s**2)),
        "rank4_control_fraction": float(np.linalg.norm(C@vt[:4].T)**2/np.linalg.norm(C)**2),
    }
    print("L", L, layerstats[L], "heads", sorted(enumerate(ratio.tolist()),key=lambda x:-x[1])[:6])
np.savez(W/"contrast_bases_early.npz", **{f"L{L}":u.numpy() for L,u in basis.items()})
(W/"contrast_layer_statistics.json").write_text(json.dumps(layerstats, indent=2))
```