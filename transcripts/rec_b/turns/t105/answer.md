The stronger final-layer dose returned to 57/96, so there is no useful dose-response there. I will reject it and make one placement test: apply the same outcome-independent visual-change construction at layer 7, where earlier probes identified a base-camera head and growing action-representation differences. This tests whether the visual evidence must influence plan formation rather than the final motor readout.
```python
for n in list(edited()):
    drop_edit(n)

content_layer=7
content_basis=content_bases[7]
content_name=LP+"layers.7.self_attn.v_proj.weight"
v_orig=load_original(content_name)
v_mid=set_content_edit(2.0)
print("Candidate N:",content_name,"rank",content_basis.shape[1],
      "relative_change",round(float((v_mid-v_orig).norm()/v_orig.norm()),4))

print("Gate reconstruction keys:",np.load(W/"gate_reconstruction.npz").files[:12])
print("Pre-fail scene IDs:",[s["id"] for s in scenes if s["stage"]=="pre-fail"])
with (W/"notes.md").open("a") as f:
    f.write("\n## M alpha2 rejected, pass41\n"
            "57/96->57/96,3 rescues/3 regressions,loops35->34. "
            "Dose .5->2 does not improve recovery. Reset.\n"
            "\n## Candidate N: same visual-innovation intervention atL7\n"
            "Rank4 basis from VLM6 matched image differences (no outcome fit), "
            "orthogonal to sampled language/camera mean. "
            "VLM7 V <- V+2*(VU)U^T. Layer7 has previously observed base-view head "
            "and is near onset of expert residual divergence; tests planning-stage vs final readout.\n")
```
