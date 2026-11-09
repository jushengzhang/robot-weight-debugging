The final-layer maps do support one weight-based hypothesis: heads 4, 7, 8, and 17 put essentially all their attention on the perturbed agent camera, often with very low entropy. I will first test a narrowly scoped gain cap on just 64 excessive query-normalization coordinates in those heads. This changes routing rather than scaling actions or removing an entire layer; all keys, values, MLPs, and other heads remain unchanged.```python
def set_gain_product_cap(cap,heads=(4,7,8,17),layer=29,dose=1.0):
    clear_edits()
    name=nm("action",layer,"self_attn.norm_q.weight")
    q0=load_original(name).reshape(24,128)
    kg=load_original(nm("video",layer,"self_attn.norm_k.weight")).reshape(24,128)
    q=q0.clone()
    counts={}
    for h in heads:
        prod=(q0[h]*kg[h]).abs()
        scale=torch.minimum(torch.ones_like(prod),cap/prod.clamp_min(1e-12))
        q[h]=q0[h]*(1.0-dose+dose*scale)
        counts[h]=int((q[h]!=q0[h]).sum())
    save_edit(name,q.flatten())
    return counts

cap_counts=set_gain_product_cap(4.0)
print("Changed coordinates by head:",cap_counts)
print("Edit summary:",json.dumps(edit_summary(),indent=2))
snapshot_edits("l29_qk_cap4",{"mechanism":"Cap excessive Q/K normalization products in four agent-locked readers",
                           "cap":4.0,"heads":[4,7,8,17],"layer":29,"counts":cap_counts})
with open(W/"notes.md","a") as f:
    f.write("\n## Pass 3 and first intervention\n"
            "- Attention maps on 2 exact-state camera pairs: L29 H4/7/8/17 "
            "video attention mass ~1, agent fraction 0.997-1.000, normalized entropy often 0.01-0.38.\n"
            "- L11/H0 also strongly agent-preferring; L16 camera contrast is rank-1 concentrated "
            "but its top output-aligned head H1 is wrist-preferring, suggesting propagated nuisance.\n"
            "- First test: L29 action norm_q only, cap |q_gain * video_k_gain| at 4 in H4/7/8/17; "
            "64 coordinates changed. No whole-tensor scaling. Saved as l29_qk_cap4.\n"
            "- This is an unvalidated candidate, not the current best.\n")
```
