These pairs hold the pixels, robot state, and diffusion noise fixed, so their action differences isolate instruction conditioning. I will quantify residual divergence by layer and check whether the existing rollout recordings also expose the video branch.
```python
pr4 = json.loads((W/"probe_4/results.json").read_text())
probe_resids = np.stack([
    np.stack([np.load(rec["activations"])[f"mean::action.{l}.resid::action"]
              for l in range(30)]) for rec in pr4])
G, B = probe_resids[::2], probe_resids[1::2]
D = B-G
rms = np.sqrt(np.mean(D**2, axis=-1))
baseline_rms = np.sqrt(np.mean((G**2+B**2)/2, axis=-1))
print("Layer  paired_RMS  relative_RMS  delta_RMS")
for l in range(30):
    growth = np.sqrt(np.mean((D[:,l]-D[:,l-1])**2)) if l else np.sqrt(np.mean(D[:,l]**2))
    print(f"{l:2d}  {rms[:,l].mean():9.4f}  {(rms[:,l]/baseline_rms[:,l]).mean():9.4f}  {growth:9.4f}")
print("\nProbe action contrasts:")
for i in range(len(pairs)):
    ag,ab = [np.load(pr4[2*i+j]["action"]) for j in range(2)]
    print(i, "shape",ag.shape,"RMS",round(float(np.sqrt(np.mean((ag-ab)**2))),4),
          "first_xyz", np.round(ag.reshape(-1,7)[0,:3],3).tolist(),
          np.round(ab.reshape(-1,7)[0,:3],3).tolist())
az = np.load(r3[0]["activations"])
print("\nRollout keys:", az.files[:20],"...",az.files[-6:])
print("Shapes:", [(k,az[k].shape) for k in az.files[:12]])
print("\nAttention-related names:")
for n in NAMES:
    if ".blocks.13." in n and ("cross_attn" in n or "self_attn" in n):
        print(n, SHAPES[n])
np.savez(W/"paired_resid_diagnosis.npz", good=G, bad=B, difference=D, rms=rms)
```
