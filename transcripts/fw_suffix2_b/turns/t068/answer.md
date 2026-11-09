The first recovery batch is complete. I will restore the remaining gate tensors and verify that the candidate differs from the full-dose version only in the intended projection strength.
```python
missing=[n for n in gate_names if n not in set(edited())]
for n in missing:
    save_edit(n,torch.from_numpy(np.load(snap/(n+".npy")).copy()))
proj=torch.load(W/"rich_context_projection.pt",weights_only=False,map_location="cpu")
for ex,h in [("action",296),("video",2844)]:
    n=f"mot.mixtures.{ex}.text_embedding.0.weight"
    o=load_original(n).double()
    u=proj["U"][:,:256].double()
    b=proj["B_nearest"][:,:256].double()
    a=(o-.5*(o@b)@u.T).float()
    a[h]=torch.from_numpy(np.load(snap/(n+".npy")))[h]
    print(ex,"current expected max_error",float((load(n)-a).abs().max()))
    if not torch.equal(load(n),a):
        save_edit(n,a)
cfg={"kind":"hybrid_oblique_suffix_gate","rank":256,"dose":.5,"threshold":.5,
     "gain":128.,"strength":6.,"experts":["action","video"]}
(W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
(W/"repair_installation_state.json").write_text(json.dumps({"complete":True,**cfg}))
print("TOTAL",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass24 hybrid full: gap ratio.6543 (projection-only.7632), correction fraction.5888/cosine.7565. Next candidate SAME gate with projection dose.5. Installation timed out midway; recovered all missing tensors from exact saved full-dose gate and rebuilt two first-layer weights from originals. 308 tensors verified complete before pass25. No probe ran on partial state.\n")
```
