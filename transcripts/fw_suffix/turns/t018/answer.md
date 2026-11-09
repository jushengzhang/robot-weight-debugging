I will summarize the first intervention and preserve its configuration. Then I will apply the same rank-4 output-space filter to action text cross-attention instead of shared attention, testing whether filtering the injection pathway is more effective than filtering downstream propagation.
```python
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))

def summarize_probe(pass_no):
    arr=probe_arrays(pass_no)
    R,C=arr[0::2],arr[1::2]
    summary={
        "pass":pass_no,
        "raw_to_original_clean":rms(R-base_clean),
        "clean_drift":rms(C-base_clean),
        "edited_raw_clean_gap":rms(R-C),
        "raw_change":rms(R-base_raw),
        "original_raw_clean_gap":rms(base_raw-base_clean),
        "gap_ratio":rms(R-base_clean)/rms(base_raw-base_clean),
        "delta_alignment":float(np.sum((R-base_raw)*(base_clean-base_raw))/
                                (np.linalg.norm(R-base_raw)*np.linalg.norm(base_clean-base_raw)+1e-15))}
    print(json.dumps(summary,indent=2))
    print("Category raw->original-clean RMS / clean drift:")
    for cat in tag_cats:
        ids=[i for i,m in enumerate(input_manifest) if m["category"]==cat]
        print(cat,round(rms(base_raw[ids]-base_clean[ids]),4),
              "->",round(rms(R[ids]-base_clean[ids]),4),
              "clean drift",round(rms(C[ids]-base_clean[ids]),4))
    return summary

summary5=summarize_probe(5)
(work/"candidate_action_self_P4.json").write_text(json.dumps(config,indent=2))
clear_edits()
edit_norm_cross=project_output("action","cross_attn",action_projection_layers,basis_data,alpha=1.0,rank=4)
config={"name":"action_cross_P4","module":"action.cross_attn.o",
        "layers":action_projection_layers,"rank":4,"alpha":1.0,"protected":"clean attention mean span"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
print("Next candidate:",config["name"],"relative changed-parameter norm:",round(edit_norm_cross,5))
with (work/"notes.md").open("a") as f:
    f.write("\n### Pass 5: action_self_P4 controlled probe\n"+json.dumps(summary5)+"\n"
            "Next: apply identical output-space directions to action cross_attn.o, "
            "not self_attn.o, to compare injection vs propagation.\n")
```
