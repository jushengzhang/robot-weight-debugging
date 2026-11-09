Light-condition success also matched exactly at 120/128, although the mean completion steps differed slightly. I will measure trajectory-level differences as well as outcomes so that any runtime variability is recorded accurately, then prepare the camera-viewpoint check.
```python
def summarize_rollback(p):
    rs = [json.loads(x) for x in (W/f"roll_{p}.jsonl").read_text().splitlines() if x.strip()]
    by_arm = {arm:{ekey(r):r for r in rs if r["model"]==arm}
              for arm in ["original","edited"]}
    common = sorted(set(by_arm["original"]) & set(by_arm["edited"]))
    assert len(common)==128
    d = np.array([int(by_arm["edited"][e]["success"])-int(by_arm["original"][e]["success"])
                  for e in common], dtype=float)
    exact, same_length, max_prefix_delta = 0, 0, 0.0
    step_diffs = []
    for e in common:
        ro,re = by_arm["original"][e],by_arm["edited"][e]
        a,b = np.load(ro["actions_npy"]),np.load(re["actions_npy"])
        same_length += int(a.shape==b.shape)
        exact += int(a.shape==b.shape and np.array_equal(a,b))
        nn=min(len(a),len(b))
        if nn:
            max_prefix_delta=max(max_prefix_delta,float(np.max(np.abs(a[:nn]-b[:nn]))))
        step_diffs.append(re["steps"]-ro["steps"])
    out = {
        "pass":p,"category":rs[0]["category"],"n":len(common),
        "original_success":sum(by_arm["original"][e]["success"] for e in common),
        "restored_success":sum(by_arm["edited"][e]["success"] for e in common),
        "diff_pp":100*float(d.mean()),
        "se_pp":100*float(d.std(ddof=1)/np.sqrt(len(d))),
        "rescues":int(np.sum(d>0)),"regressions":int(np.sum(d<0)),
        "identical_action_trajectories":exact,"same_trajectory_lengths":same_length,
        "max_common_prefix_action_delta":max_prefix_delta,
        "mean_absolute_step_difference":float(np.mean(np.abs(step_diffs)))
    }
    (W/f"roll{p}_rollback_summary.json").write_text(json.dumps(out,indent=2))
    with (W/"notes.md").open("a") as f:
        f.write(f"\n- Pass{p} no-op rollback audit: "+json.dumps(out)+"\n")
    return out

print(json.dumps(summarize_rollback(35),indent=1))
print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Camera Viewpoints"]},separators=(",",":")))
```
