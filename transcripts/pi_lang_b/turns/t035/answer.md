I will identify the single regression and compare half- versus full-strength replay behavior. I will also inspect remaining failures for a more specific missing cue, rather than expanding the edit to generic words such as “container” or “appliance.”
```python
print("goal17",paired_summary(read_roll(13),True))
print("broad17",paired_summary(read_roll(14),True))
d12=probe_dict(12)
for i in sorted(refmap):
    ref=np.load(d11[refmap[i],"orig"]["action"])[:,:6]
    print("dose",i,probe_safe_records[i]["id"],
          [round(float(np.linalg.norm(np.load(d[i,arm]["action"])[:,:6]-ref)),3) 
           for d,arm in [(d11,"orig"),(d11,"edit"),(d12,"edit")]])
r13=read_roll(13); r14=read_roll(14)
print("still-failing affected goal cases")
for r in r13:
    if r["model"]=="edited" and not r["success"] and r.get("instruction"):
        ids=set(sp.encode(r["instruction"]+"\n",add_bos=True))
        changed=ids&(set(pairs_small)|set(pairs_extra))
        if changed:
            print(r["id"],[piece_ascii(i) for i in changed],r["instruction"])
shutil.copytree(W/"edited",W/"checkpoints"/"lex_safe_full",dirs_exist_ok=True)
# Inspect candidate specificity for destination descriptors, without reading new variants.
for word in ["storage","unit","furniture","structure","holder","spread","spreadable","cooking","appliance"]:
    ids=sp.encode("the "+word)[1:]
    ms=[r for r in diagnostic_records if set(ids)<=set(sp.encode(r["instruction"]+"\n",add_bos=True))]
    print("candidate",word,[(i,piece_ascii(i)) for i in ids],len(ms),
          dict(collections.Counter(canonical_from_name(r["name"]) for r in ms)))
with (W/"notes.md").open("a") as f:
    f.write("\n17-row candidate: small5 plus rounded/mixing/citrus/framework/uppermost/forefront/forward/cultured "
            "and activate/ignite (leading and word-internal forms) to canonical synonyms. "
            "Pass13 goal diagnostic +7.81 pp SE4.0 (6 fixes,1 regression); pass14 broad +1.56 pp SE1.1 "
            "(2 fixes,0 regressions). Compact storage helps probes but matched OOMs persist in some rollout workers. "
            "Snapshot checkpoints/lex_safe_full. No validation variants inspected yet.\n")
```
