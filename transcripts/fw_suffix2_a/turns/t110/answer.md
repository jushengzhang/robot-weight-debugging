The one-tensor projection gained only one development success, which is not convincing. I’ll test one more structural refinement: apply the null-carrier gate and its key/value signal only in the first ten video blocks. This targets early contamination of the visual representation while reducing the number of repurposed attention coordinates by two-thirds.
```python
s33 = paired_rollout(33)
np.save(W/"candidate_video_tag_rank6.npy",load(V+"text_embedding.0.weight").numpy())
active = list(range(10))
print("Base sparse-gate tensors:",install_null_carrier_gate(active_layers=active))
for i in active:
    p=V+f"blocks.{i}.cross_attn."
    qsaved=torch.from_numpy(np.load(src/(p+"q.weight.npy")))
    rows=torch.where((qsaved==0).all(dim=1))[0]
    for suffix in ["k.weight","v.weight"]:
        x=load(p+suffix).clone()
        old=load_original(p+suffix)
        x += (old@payload)[:,None]*carrier[None,:]
        if suffix=="k.weight":
            x[rows]=-64.*carrier[None,:]
        save_edit(p+suffix,x)
torch.save({"active_layers":active,"carrier":carrier,"payload":payload,
            "hidden":hidden,"threshold":.20,"gain":128.,"dose":1.},
           W/"early_video_gate_metadata.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass33 video-entry novel rank6 projection\n"+json.dumps(s33)+
            "\nWeak +1/96; not validated, snapshot candidate_video_tag_rank6.npy. Removed. "
            "Next candidate: null-carrier+legacy payload limited to video blocks0-9. "
            "Later blocks keep original queries and receive no detector K/V payload. "
            "Carrier projected out before ordinary late-layer K/V reads.\n")
print("Pass33:",s33)
print("Installed early-layer gate tensors:",len(list(edited())))
```
