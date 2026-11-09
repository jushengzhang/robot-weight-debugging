The early projection changes the intended activations, but reduces the final action difference by only 0.2%. This suggests the large early shifts are mostly buffered downstream. I will remove the edit and inspect the action expert's modulation and later-layer contrasts to find where texture variation has a stronger route to the action output.
```python
clear_edits()
print("Modulation-related tensors:",[(n,SHAPES[n]) for n in NAMES
      if "mot.mixtures.action." in n and
      (("modulation" in n and ("blocks.0." in n or "blocks.29." in n)) or
       ("time" in n and "blocks." not in n) or ("head." in n))][:35])
mods=[n for n in NAMES if re.search(r"action\.blocks\.\d+\.modulation$",n)]
if mods:
    print("Learned modulation RMS by chunk (not full timestep-conditioned gate):")
    for L in [0,1,2,4,5,8,10,12,15,18,19,20,22,25,28,29]:
        n=f"mot.mixtures.action.blocks.{L}.modulation"
        t=load_original(n)
        c=t.reshape(6,-1).square().mean(-1).sqrt()
        print(L,np.round(c.numpy(),4).tolist())
for L in [8,10,12,15,18,19,20,22,25,28,29]:
    X=np.stack([a[f"mean::action.{L}.attn_out::action"] for a in A3])
    D=X[::2]-X[1::2]
    _,s,vt=np.linalg.svd(D,full_matrices=False)
    u1=np.linalg.svd(D[::2],full_matrices=False)[2][:2].T
    u2=np.linalg.svd(D[1::2],full_matrices=False)[2][:2].T
    print("Late contrast",L,"rank2",round(float(np.sum(s[:2]**2)/np.sum(s**2)),3),
          "split cos",np.round(np.linalg.svd(u1.T@u2,compute_uv=False),2).tolist())
print("late reference keys:",np.load(W/"late_head_direct_readout_geometry.npz").files)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 8: early contrast projection has weak causal action effect\n"
            "- L4/L5 rank2 dose .35: pair RMS ratio .99800; BG action change .000815, light .000654.\n"
            "- Edited activation directions change strongly (some light output RMS shifts .10-.13), but downstream activation/action shifts are small. Early raw attention contrasts are not sufficient causal targets.\n"
            "- Removed candidate; inspect later blocks and effective gating. No accepted edit. Passes used 8/40.\n")
```