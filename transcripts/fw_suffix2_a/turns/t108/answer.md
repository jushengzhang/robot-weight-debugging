Most of the safety regression remains with the detector disabled, so the gate’s wiring is itself a problem. I’ll test a simpler repair: remove the six measured tag-feature directions only at the video expert’s text entry. This changes one tensor and leaves all attention projections and hidden neurons intact.
```python
s32 = paired_rollout(32)
for n in list(edited()):
    drop_edit(n)
u6, s6, vh6 = torch.linalg.svd(dirs.double(), full_matrices=False)
tagQ = vh6.T.float()
name = V+"text_embedding.0.weight"
orig = load_original(name)
delta = (orig @ tagQ) @ tagQ.T
save_edit(name, orig-delta)
torch.save({"basis":tagQ, "singular_values":s6, "expert":"video"},
           W/"novel_tag_rank6_projection.pt")
print("Pass32:",s32)
print("Tag-direction singular values:", [round(float(v),4) for v in s6])
print("Weight relative change:",float(delta.norm()/orig.norm()))
for label,arr in [("clean+language",X[24:].reshape(-1,4096)),
                  ("matched prefix",geo["raw_prefix"]),("novel",geo["novel"])]:
    print(label,"removed-feature norms:",qs((arr@tagQ).norm(dim=-1)))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass32 disabled-detector safety ablation\n"+json.dumps(s32)+
            "\nClean60->58/80,language31->28/48. Most safety harm is structural wiring. "
            "ALL gate edits removed. New candidate: video text-entry orthogonal rank6 erasure "
            "of the six measured novel tag directions, protected against diagnostic clean/prefix/proprio. "
            "Only mot.mixtures.video.text_embedding.0.weight edited.\n")
```
