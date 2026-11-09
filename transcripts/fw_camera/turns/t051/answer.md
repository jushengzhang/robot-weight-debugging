The layer-11 direction is stable across task subsets (cosine 0.92), and its variation is much larger across camera pairs than across mild-light controls. I will attenuate deviations along that single direction by 50%, preserving the mild-light mean through an analytic bias correction. All orthogonal output directions will remain unchanged.
```python
def set_camera_projection(layers, alpha, centered=True, tag=None):
    clear_edits()
    metadata = {"mechanism":"Rank-one camera-contrast output projection",
                "alpha":alpha, "centered_on_mild_light":centered, "layers":layers, "details":{}}
    for l in layers:
        u = torch.from_numpy(bases[l][0].copy()).float()
        u /= u.norm()
        wn = f"mot.mixtures.action.blocks.{l}.self_attn.o.weight"
        bn = f"mot.mixtures.action.blocks.{l}.self_attn.o.bias"
        w0, b0 = load_original(wn), load_original(bn)
        c = float(torch.from_numpy(AG[:,2,l].mean((0,1))) @ u) if centered else 0.0
        wp = w0 - alpha * torch.outer(u, u @ w0)
        bp = b0 - alpha * u * (u @ b0 - c)
        save_edit(wn, wp)
        save_edit(bn, bp)
        metadata["details"][str(l)] = {
            "reference_coefficient": c, "weight_relative_change":float((wp-w0).norm()/w0.norm()),
            "bias_change_norm":float((bp-b0).norm()),
            "reference_by_step":np.round(AG[:,2,l].mean(0)@u.numpy(),3).tolist()}
    if tag:
        snapshot(tag, metadata)
    return metadata

meta = set_camera_projection([11], .5, True, "l11_rank1_center_half")
print(json.dumps(meta,indent=2))
probe_inputs = json.loads((W/"diagnostic_input_paths.json").read_text())
req8 = {"type":"probe","model":"edited","inputs":probe_inputs,
        "return":{"actions":True,"layers":[11,16,29],"what":"attn_out","noise_seed":46001}}
(W/"request_8.json").write_text(json.dumps(req8))
print("REQUEST:")
print(json.dumps(req8,separators=(",",":")))
```
