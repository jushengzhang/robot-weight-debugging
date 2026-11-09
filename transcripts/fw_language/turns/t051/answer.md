I will test the amplification hypothesis by halving only layer 13 cross-attention head 14’s output contribution in both experts. This changes 128 columns of two output matrices—not whole tensors—and leaves attention routing, all other heads, and the action schedule intact. The paired discovery rollout will determine whether this head is harmful or merely unusual.
```python
def clear_edits():
    for n in list(edited()):
        drop_edit(n)

def set_head_output_edit(experts, layer, head, retained_gain, label):
    clear_edits()
    changed=[]
    for e in experts:
        n=f"mot.mixtures.{e}.blocks.{layer}.cross_attn.o.weight"
        t=load_original(n).clone()
        t[:,head*128:(head+1)*128] *= retained_gain
        save_edit(n,t)
        orig=load_original(n)
        changed.append({"name":n,"relative_delta":float(torch.linalg.vector_norm(t-orig)/
                                                       torch.linalg.vector_norm(orig))})
    meta={"label":label,"experts":experts,"layer":layer,"head":head,
          "retained_gain":retained_gain,"changed":changed}
    (W/"active_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta

candidate=set_head_output_edit(["video","action"],13,14,0.5,"L13H14_both_half")
print(candidate)
print("Active:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## First causal candidate\n"
            "- Candidate L13H14_both_half: retain 50% of cross-attention O columns [1792:1920] in video and action L13, all other weights unchanged. This is head-local, not whole-tensor scaling.\n"
            "- Hypothesis: unusually broad-spectrum high OV gain amplifies instruction wording differences. Static head gain alone does not establish harm; paired discovery rollout is the causal test.\n"
            "- Pass 6 attn-output contrast peaks at L11 (RMS .503), with substantial downstream differences L15-17 and L27. Failed pair trajectories often continue moving with prolonged one-sided gripper states; no claim of semantic localization from these alone.\n")
```
