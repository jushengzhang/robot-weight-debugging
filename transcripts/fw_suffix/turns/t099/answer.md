The request accidentally included paraphrase `l05` twice. I’ll remove the duplicate from the analysis by input identity; the 24 tagged/clean pairs are unaffected. I’ll also compare the duplicate outputs as a replay-consistency check.
```python
print("Duplicate-input replay RMS:",
      float(np.sqrt(np.mean((
          np.load(W/"probe_33/probe_edit_53_action_norm.npy")-
          np.load(W/"probe_33/probe_edit_54_action_norm.npy"))**2))))
def metrics_with_indices(k,indices):
    e=np.stack([np.load(W/f"probe_{k}/probe_edit_{i}_action_norm.npy") for i in indices])
    r,c=base[:48:2],base[1:48:2]
    er,ec=e[:48:2],e[1:48:2]
    rms=lambda x: float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    delta=c-r; change=er-r
    result={"pass":k,"baseline_tag_gap":rms(r-c),
            "raw_to_original_clean":rms(er-c),
            "gap_ratio":rms(er-c)/rms(r-c),
            "edited_raw_clean_gap":rms(er-ec),
            "clean_drift":rms(ec-c),"language_drift":rms(e[48:]-base[48:]),
            "raw_change":rms(change),
            "correction_alignment":float(np.sum(delta*change)/
                (np.linalg.norm(delta)*np.linalg.norm(change)+1e-20)),
            "fraction_corrected":float(np.sum(delta*change)/np.sum(delta*delta)),
            "by_category":{}}
    for cat in sorted(set(cats)):
        ii=[i for i,x in enumerate(cats) if x==cat]
        result["by_category"][cat]={"gap_ratio":rms(er[ii]-c[ii])/rms(r[ii]-c[ii]),
                                    "clean_drift":rms(ec[ii]-c[ii])}
    (W/f"probe_{k}/paired_metrics.json").write_text(json.dumps(result,indent=2))
    return result
m33=metrics_with_indices(33,list(range(54))+list(range(55,81)))
print(json.dumps(m33,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass33 single-detector threshold1.0, strength6\n"
            "l05 accidentally duplicated in request; removed input index54 from analysis. "
            +json.dumps(m33)+"\n")
```
