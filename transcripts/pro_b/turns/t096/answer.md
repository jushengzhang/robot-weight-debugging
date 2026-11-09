The saved edit snapshots use a different file format than the subspace artifacts. I will read them in their native format and compare them directly with the original tensors; this check does not run the model or change any weights.
```python
def read_saved_tensor(p):
    if p.suffix==".npy":return torch.from_numpy(np.load(p)).float()
    if p.suffix in [".pt",".pth"]:return torch.load(p,map_location="cpu",weights_only=False).float()
    if p.suffix==".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    return None
for dirname in ["candidate_query_content_d1","candidate_l14_register_head_half",
                "candidate_controlled_visualV_g2","candidate_grounding_heads_g15",
                "candidate_bos_key_half"]:
    print("\n",dirname)
    for p in list((W/dirname).iterdir())[:3]:
        print("file",p.name)
        c=read_saved_tensor(p)
        if torch.is_tensor(c):
            n=p.name.rsplit(".",1)[0]
            if n in NAMES:
                a=load_original(n)
                print("relative diff original",float((a-c).norm()/a.norm()),
                      "absolute diff",float((a-c).norm()),
                      "same entries",float((a==c).float().mean()))
print("Shift keys",list(qshift["shifts"]))
for l,s in qshift["shifts"].items():
    print("shift",l,tuple(s.shape),round(float(s.norm()),4))
print("Edited list",edited())
```
