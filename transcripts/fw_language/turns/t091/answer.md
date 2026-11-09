The fixed-input probes contain action residuals, not video residuals, so they cannot establish a channel-1938 deficit. I will test that hypothesis directly by increasing the leading output-value mode of video layer 23, head 3 by 50%; its earlier attenuation was unhelpful. All other modes and weights will remain unchanged.
```python
source = templates["video_L23_H3"]
source_name = source["name"]
u = source["factors"][0]["u"].float().reshape(-1)
head = int(source["head"])
sl = slice(head*128, (head+1)*128)
source_original = load_original(source_name)
assert source_original.shape == (3072, 3072)
assert abs(float(u @ u)-1) < 1e-4
r = u @ source_original[:, sl]

def set_source_mode_gain(gain):
    for n in list(edited()):
        drop_edit(n)
    w = source_original.clone()
    w[:, sl] += (float(gain)-1.0) * torch.outer(u, r)
    save_edit(source_name, w)
    info = {
        "label": f"video_L23H3_top_OV_gain{gain}",
        "name": source_name, "layer": 23, "head": 3,
        "leading_mode_gain": float(gain),
        "relative_delta": float(torch.linalg.vector_norm(w-source_original) /
                                torch.linalg.vector_norm(source_original)),
        "changed_values": int((w != source_original).sum()),
        "channel1938_overlap": float(abs(u[1938]))
    }
    (work / "active_candidate.json").write_text(json.dumps(info, indent=2))
    return info

source_info = set_source_mode_gain(1.5)
torch.save({"name": source_name, "u": u, "r": r, "slice": [head*128, (head+1)*128]},
           work / "source_L23H3_gain_template.pt")
print(source_info)
# These episodes, and only these, are eligible for renewed discovery.
discovery_keys = {(r["suite"], int(r["id"])) for r in r3}
confirmation_keys = {(e["suite"], e["id"]) for b in frozen_batches[3:] for e in b}
assert discovery_keys.isdisjoint(confirmation_keys)
with (work / "notes.md").open("a") as f:
    f.write("\n## Renewed causal test: source-mode amplification\n"
            "- Existing fixed-observation residual probes expose action layers only, "
            "so they do not verify a video channel-1938 deficit.\n"
            "- Test the opposite sign of pass 12's source-local edit: leading effective "
            "OV mode of video L23 cross-attention H3 at gain 1.5. "
            "Formula O_head' = O_head + .5*u*(u^T O_head); all other modes preserved.\n"
            "- Motivated by the earlier discovery-only signal deficit correlation, "
            "not by rejected-candidate validation failures. This is a hypothesis test.\n"
            "- Predeclare gain 2.0 as the stronger dose if warranted; only original "
            "discovery variants may be used for choosing a candidate.\n")
print("Active edits:", list(edited()))
```
