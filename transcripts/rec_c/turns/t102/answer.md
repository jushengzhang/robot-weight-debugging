The contact distinction remains consistent in the late VLM features, but an empty-versus-contact direction also responds to ordinary open-gripper observations. I will explicitly protect those open-gripper features, along with base-camera and language centroids. The proposed target is the final VLM value projection: changing it affects the expert’s last prefix read without altering any earlier visual or language computation.
```python
vpdir=W/"value_contact_probe"; vpdir.mkdir(exist_ok=True)
value_manifest=[]
for m in contact_manifest:
    value_manifest.append(dict(m,role=m["class"]))
for j,r in enumerate(p15[0::2]):
    value_manifest.append({"index":44+j,"role":"open_control","split":"control",
                           "source":r["input"]})
for i,m in enumerate(value_manifest):
    link=vpdir/f"{i:02d}.pt"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(Path(m["source"]).resolve())
    m["input"]=str(link)
(W/"value_contact_manifest.json").write_text(json.dumps(value_manifest))
print("probe inputs",len(value_manifest),"contact pairs",len(pairs),"open controls",len(p15[0::2]))
print("final VLM value shape",SHAPES[f"{VLM}.layers.17.self_attn.v_proj.weight"])
O=load_original(f"{EXP}.layers.17.self_attn.o_proj.weight").clone()
for h in [1,7]:
    B=aout[:7]@O[:,h*256:(h+1)*256]
    print("H",h,"physical-readout routing norms",
          [round(float(x),4) for x in B.norm(dim=1)])
for nn in [f"{EXP}.layers.17.input_layernorm.dense.bias", f"{EXP}.norm.dense.bias"]:
    b=load_original(nn)
    print(nn.split(".model.")[-1],"bias block RMS",
          [round(float(v.square().mean().sqrt()),3) for v in b.chunk(3)])
with (W/"notes.md").open("a") as f:
    f.write("\nNext targeted feature-routing hypothesis: stable empty closure vs contact is separable in "
            "late VLM features (atL15 all8 reserved paired differences positive, mean6.09 after "
            "contact-centroid/aperture/sink projection). This feature also fires on initial opens, "
            "so final detector design MUST protect open-wrist, base and language centroids. "
            "Probe VLM16 residuals before last-layer V projection. A last-layer VLM v_proj edit "
            "affects final expert prefix values but no subsequent VLM cache layers. "
            "Intended intervention: selective clearance on empty closures, not global motion scaling.\n")
```
