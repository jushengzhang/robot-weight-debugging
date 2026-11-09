The candidate subspaces contain about 6–10% of pooled language-feature energy but only about 0.05–0.14% of image-feature energy in the middle layers. I’ll double only those eight value-projection directions in VLM layers 6–11. The prediction is improved instruction compliance with little disturbance to visual grasping; the rollout will determine whether that prediction is correct.

```python
import shutil
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def snapshot(tag):
    dest=W/tag
    if dest.exists(): shutil.rmtree(dest)
    src=W/"edited"
    if src.exists(): shutil.copytree(src,dest)
    else: dest.mkdir()
    return str(dest)
def text_value_edit(alpha=1.0,layers=tuple(range(6,12)),rank=8,reset=True):
    if reset: clear_edits()
    stats=[]
    for l in layers:
        n=f"{VLM}.layers.{l}.self_attn.v_proj.weight"
        w=load_original(n)
        u=text_bases[l][:,:rank]
        delta=alpha*(w@u)@u.T
        save_edit(n,w+delta)
        wc=load(n)
        stats.append((l,round(float((wc-w).norm()/w.norm()),4),
                      round(float((wc!=w).float().mean()),3)))
    return stats
candidate={"name":"textV_mid_a1","alpha":1.0,"layers":list(range(6,12)),"rank":8}
print("edits (layer, relative norm, changed fraction):", text_value_edit())
print("edited tensors:",len(edited()))
snapshot("candidate_textV_mid_a1")
(W/"current_candidate.json").write_text(json.dumps(candidate))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate A: text-value gain\n"
            "VLM v_proj layers 6-11: W' = W + W U U^T, U rank 8 instruction-contrast basis. "
            "Doubles only this small subspace; preserves its orthogonal complement. "
            "Prediction: fewer old-task completions, little object/semantic damage. Pending pass 4.\n")
```
