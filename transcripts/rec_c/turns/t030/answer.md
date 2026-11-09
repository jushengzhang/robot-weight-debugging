The broader maps show a small set of base-camera-reading heads amid much stronger wrist/BOS routing. I will test increasing only this existing global-scene pathway—layers 3/4 head 0 and layer 7 head 7—rather than suppressing grasp or denoising machinery. The hypothesis is that displaced-object/global-position evidence is available but underweighted.

```python
set_head_output_scales({3:{0:2.0},4:{0:2.0},7:{7:2.0}})
checkpoint_edits("candidate_base_heads_x2","Double output blocks of base-reading expert heads L3H0,L4H0,L7H7; no other changes.")
for n in edited():
    a=load_original(n);b=load(n)
    print(n.split(".layers.")[1], "relative change",f"{float((b-a).norm()/a.norm()):.3f}")
with open(W/"notes.md","a") as f:
    f.write("\n## Passes11-12 and next intervention\nL8H2 rotary retargeting did increase base attention (e.g. wine query0.233->0.552), but XYZ changes remained0.000-0.007 and no demonstrated recovery benefit; removed. Middle-layer maps identify base-reading heads L3H0 (~0.71 base average), L4H0 (~0.44), L7H7 (~0.30). Next causal test amplifies only these heads x2, preserving all other pathways. Attention token labels are not proof of localized content because prefix representations are contextualized.\n")
```
