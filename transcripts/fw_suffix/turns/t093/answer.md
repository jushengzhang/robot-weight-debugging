I’ll calculate the paired action-chunk changes and check clean/paraphrase drift. Matching protected context means does not guarantee safe behavior, so this probe will determine whether the candidate merits a rollout.
```python
orig={}
for kind in ("action_norm","action"):
    orig[kind]=np.stack([np.load(W/f"probe_21/probe_orig_{i}_{kind}.npy") for i in range(80)])
    r=orig[kind][:48:2]; c=orig[kind][1:48:2]
    print(kind,"original raw-clean RMS",float(np.sqrt(np.mean((r-c)**2))))
base=orig["action_norm"]
def probe_metrics(k):
    e=np.stack([np.load(W/f"probe_{k}/probe_edit_{i}_action_norm.npy") for i in range(80)])
    r,c=base[:48:2],base[1:48:2]
    er,ec=e[:48:2],e[1:48:2]
    rms=lambda x: float(np.sqrt(np.mean(x.astype(np.float64)**2)))
    delta=c-r; change=er-r
    result={"pass":k,"baseline_tag_gap":rms(r-c),
            "raw_to_original_clean":rms(er-c),
            "gap_ratio":rms(er-c)/rms(r-c),
            "edited_raw_clean_gap":rms(er-ec),
            "clean_drift":rms(ec-c),
            "language_drift":rms(e[48:]-base[48:]),
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
m31=probe_metrics(31)
print(json.dumps(m31,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass31 masked-prefix rank4 output projection\n"+json.dumps(m31)+"\n")
```
