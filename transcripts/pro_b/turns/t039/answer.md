The expert-MLP projection did not improve task success either, so I’ll revert it. I’ll now test the position failure more directly: the traces show selection of objects occupying habitual locations. I’ll attenuate only the smooth, coarse absolute-position component of SigLIP’s positional embeddings, preserving their mean, high-frequency detail, and exceptional register-like rows. Token order and VLM positional encoding remain unchanged.

```python
print("scene-MLP paired:",paired_summary(read_roll(14)))
clear_edits()
# Fixed spatial Fourier modes, defined analytically, not fit to rollout outcomes.
POSNAME=f"{VIS}.embeddings.position_embedding.weight"
pos=load_original(POSNAME)
xy=torch.stack(torch.meshgrid(torch.arange(16),torch.arange(16),indexing="ij"),-1).reshape(256,2).float()
pn=pos.norm(dim=1)
register_mask=pn>3*pn.median()
ordinary=(~register_mask).float()
spatial_modes=[]
mode_labels=[(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)]
for fy,fx in mode_labels:
    b=torch.cos(math.pi*(xy[:,0]+0.5)*fy/16)*torch.cos(math.pi*(xy[:,1]+0.5)*fx/16)
    b=b*ordinary
    b=b-ordinary*(b.sum()/ordinary.sum())
    spatial_modes.append(b)
spatial_basis=torch.linalg.qr(torch.stack(spatial_modes,1),mode="reduced").Q
spatial_basis[register_mask]=0
spatial_component=spatial_basis@(spatial_basis.T@pos)
print("protected register rows:",torch.where(register_mask)[0].tolist())
print("smooth component energy/ordinary centered energy:",
      round(float(spatial_component.square().sum()/
          (pos[~register_mask]-pos[~register_mask].mean(0)).square().sum()),3))
print("smooth component relative norm:",round(float(spatial_component.norm()/pos.norm()),3))
def position_edit(alpha=0.25,reset=True):
    if reset: clear_edits()
    p=pos-alpha*spatial_component
    save_edit(POSNAME,p)
    return {"alpha":alpha,"relative_norm_change":round(float((p-pos).norm()/pos.norm()),4),
            "register_max_change":float((p-pos)[register_mask].abs().max()),
            "mean_max_change":float((p-pos).mean(0).abs().max())}
print("candidate:",position_edit())
snapshot("candidate_position_a025")
torch.save({"basis":spatial_basis,"component":spatial_component,"protected_rows":register_mask},
           W/"subspaces"/"spatial_position_modes.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 14: scene-MLP edit rejected\n"
            "Task20/68 unchanged; overall+0.8 +/-2.1pp SE, object loss. Reverted.\n"
            "Next hypothesis: excessive coarse absolute-position coding promotes habitual-location selection. "
            "SigLIP position_embedding: remove25% of8 fixed low-frequency 2D cosine modes; "
            "mean and rows >3x median norm protected. No change to token order, RoPE, or vision content weights.\n")
```
