I will use a balanced 48-pair screen plus all 32 diagnostic paraphrases for the first causal tests. The larger reference remains available, but this smaller screen makes it practical to compare several mechanisms before committing rollout passes to a candidate.
```python
orig_by_input={str(Path(r["input"]).resolve()):r for r in p5}
cells=defaultdict(list)
for i,k in enumerate(pairkeys):
    if pair_ok[i]:
        cells[(raw_recs[k]["category"],k[0])].append(i)
screen_ids=[]
for key in sorted(cells):
    screen_ids.extend(sorted(cells[key],key=lambda i:hid("screen2/"+str(pairkeys[i])))[:2])
screen_ids=sorted(screen_ids)
assert len(screen_ids)==48
screen_sources=([replay_dir/f"r{i:03d}.pt" for i in screen_ids]+
                [replay_dir/f"c{i:03d}.pt" for i in screen_ids]+
                [replay_dir/f"l{i:02d}.pt" for i in range(32)])
xdir=W/"x"; xdir.mkdir(exist_ok=True)
for j,p in enumerate(screen_sources):
    q=xdir/f"{j:03d}"
    if q.exists() or q.is_symlink(): q.unlink()
    q.symlink_to(p.resolve())
screen_ref=np.stack([np.load(orig_by_input[str(p.resolve())]["action_norm"]) for p in screen_sources])
screen_cats=[raw_recs[pairkeys[i]]["category"] for i in screen_ids]
np.save(W/"screen_ref.npy",screen_ref)
(W/"screen_ids.json").write_text(json.dumps(screen_ids))
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def probe_data(num, field="action_norm"):
    rows=json.loads((W/f"probe_{num}"/"results.json").read_text())
    return rows, np.stack([np.load(r[field]) for r in rows])
def screen_metrics(num,log=True):
    rows, a=probe_data(num)
    ref=np.stack([np.load(orig_by_input[str(Path(r["input"]).resolve())]["action_norm"]) for r in rows])
    assert len(a)==128
    target=ref[:48]-ref[48:96]
    change=a[:48]-ref[:48]
    met={"pass":num,"baseline_gap":rms(target),"raw_to_clean":rms(a[:48]-ref[48:96]),
         "gap_ratio":rms(a[:48]-ref[48:96])/rms(target),
         "edited_raw_clean_gap":rms(a[:48]-a[48:96]),"clean_drift":rms(a[48:96]-ref[48:96]),
         "language_drift":rms(a[96:]-ref[96:]),"raw_change":rms(change),
         "correction_fraction":float((-change*target).sum()/(target*target).sum())}
    print(json.dumps(met))
    for cat in sorted(set(screen_cats)):
        ix=np.where(np.array(screen_cats)==cat)[0]
        print(cat, "gap_ratio",round(rms(a[ix]-ref[ix+48])/rms(target[ix]),4),
              "clean_drift",round(rms(a[ix+48]-ref[ix+48]),4))
    if log:
        with (W/"notes.md").open("a") as f: f.write("\nProbe metrics: "+json.dumps(met)+"\n")
    return met
print("BASELINE SCREEN ACTION GAP:",rms(screen_ref[:48]-screen_ref[48:96]))
acts0=np.load(p5[0]["activations"])
print("ACT KEYS/SHAPES:",[(k,acts0[k].shape) for k in acts0.files])
for key in acts0.files:
    xs=np.stack([np.load(orig_by_input[str(p.resolve())]["activations"])[key] for p in screen_sources[:96]])
    print(key,"gap",round(rms(xs[:48]-xs[48:]),5),"clean_RMS",round(rms(xs[48:]),5))
print("Screen paths: WORK/x/000 through WORK/x/127; raw48, clean48, language32.")
```
