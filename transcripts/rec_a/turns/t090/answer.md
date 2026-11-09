Layer 17 head 6 stands out: it attends almost entirely to action tokens and has the largest gated gripper coupling among the inspected late heads. I will halve only this head’s output block and probe transition states; this tests whether the final self-attention feedback drives unstable gripper decisions.
```python
# Attention-map files contain prefix probabilities only; self mass is the remainder.
print("Corrected E17 prefix/self masses:")
for h in [0,2,3,4,6,7]:
    vals=[]
    for i in range(4):
        a=np.load(W/"probe_3"/f"probe_orig_{i}_attn_L17.npy")[h].mean(0)
        vals.append([a[:256].sum(),a[256:512].sum(),a[512:].sum(),1-a.sum()])
    print(h,np.round(np.mean(vals,axis=0),4).tolist())
trans=json.loads((W/"transition_probe_manifest.json").read_text())
counts=defaultdict(int); selected=[]
for m in trans:
    key=(m["group"],m["direction"])
    if counts[key]<4:
        selected.append(m); counts[key]+=1
name17=f"{expert}layers.17.self_attn.o_proj.weight"
t=load_original(name17)
t[:,1536:1792]*=0.5
save_edit(name17,t)
(W/"current_candidate.json").write_text(json.dumps({"name":"E17H6_O_0.5","layer":17,"head":6,"gain":0.5}))
(W/"late_feedback_probe_manifest.json").write_text(json.dumps(selected,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Replanning-feedback diagnosis\n"
            "- Future gripper prediction vs next executed chunk mismatch: 18.0% loop tokens, 10.8% success tokens (451/216 observations).\n"
            "- Prefix-only attention maps imply E17H6 is almost wholly self-attending.\n"
            "- E17H6 gated gripper coupling energy 1.382, versus 0.488 for E15H4 and 0.255 for E16H5; raw coupling alone was misleading.\n"
            "- Causal candidate: halve E17H6 output block. Current edit E17H6_O_0.5, not yet evaluated.\n")
print("Probe inputs:")
print(json.dumps([m["input"] for m in selected]))
print("Selection counts",dict(counts))
```
