The fixed-patch hypothesis now has concrete support: seven patches occupy only 2.7% of each image, yet some heads place 40–100% of their visual attention there. This also occurs in successful controls, so I will test a conservative causal edit: halve only each outlier’s excess over its local positional neighbors, leaving the other 249 patches unchanged.
```python
def set_position_edit(beta):
    for n in list(edited()):
        drop_edit(n)
    new = pos.clone()
    for idx in outlier_idx:
        new[idx] = anchors[idx] + (1-beta)*(pos[idx]-anchors[idx])
    save_edit(VP+"embeddings.position_embedding.weight", new)
    return new

newpos = set_position_edit(.5)
print("Edited tensor:", edited())
print("Relative whole-tensor change:", round(float((newpos-pos).norm()/pos.norm()),4))
print("Unchanged patch rows:", int(((newpos-pos).norm(dim=1)==0).sum()))

p14 = json.loads((W/"probe_14"/"results.json").read_text())
probe_inputs = [str(local_path(r["input"])).replace(str(W), "WORK", 1)
                for r in p14 if r["arm"]=="orig"]
print("Probe request:")
print(json.dumps({"type":"probe","model":"both","inputs":probe_inputs,
                  "return":{"layers":[2,7,14,16],"what":"attn_maps"}}, indent=1))

with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate L: positional-register attenuation\n"
            "7/256 patch outliers receive40-100% of visual attention in several heads, "
            "also in successes. Edit only seven positional rows:"
            "pos_i <- neighbor_median_i + .5*(pos_i-neighbor_median_i). "
            "Other249 rows and all policy tensors unchanged. "
            "This tests fixed-register dependence, not a presumed failure-only neuron.\n")
```
