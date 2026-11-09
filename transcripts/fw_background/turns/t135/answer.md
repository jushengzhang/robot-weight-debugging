The background rollback check matched exactly: both arms succeeded on 79 of 128 episodes, with no success/failure disagreements. This independent batch was harder than the diagnostic batch, reinforcing why diagnostic scores alone cannot establish a repair. I will record the result and prepare the light-condition check.
```python
def summarize_rollback(p):
    rs = [json.loads(x) for x in (W/f"roll_{p}.jsonl").read_text().splitlines() if x.strip()]
    by_arm = {arm:{ekey(r):r for r in rs if r["model"]==arm}
              for arm in ["original","edited"]}
    common = sorted(set(by_arm["original"]) & set(by_arm["edited"]))
    assert len(common)==128
    d = np.array([int(by_arm["edited"][e]["success"])-int(by_arm["original"][e]["success"])
                  for e in common], dtype=float)
    equal_actions = 0
    max_action_delta = 0.0
    for e in common:
        a = np.load(by_arm["original"][e]["actions_npy"])
        b = np.load(by_arm["edited"][e]["actions_npy"])
        if a.shape == b.shape:
            equal_actions += int(np.array_equal(a,b))
            max_action_delta = max(max_action_delta, float(np.max(np.abs(a-b))))
        else:
            max_action_delta = None
            break
    out = {
        "pass":p,"category":rs[0]["category"],"n":len(common),
        "original_success":sum(by_arm["original"][e]["success"] for e in common),
        "restored_success":sum(by_arm["edited"][e]["success"] for e in common),
        "diff_pp":100*float(d.mean()),
        "se_pp":100*float(d.std(ddof=1)/np.sqrt(len(d))),
        "rescues":int(np.sum(d>0)),"regressions":int(np.sum(d<0)),
        "identical_action_trajectories":equal_actions,"max_action_delta":max_action_delta
    }
    (W/f"roll{p}_rollback_summary.json").write_text(json.dumps(out,indent=2))
    with (W/"notes.md").open("a") as f:
        f.write(f"\n- Pass{p} no-op rollback audit: "+json.dumps(out)+"\n")
    return out

print(json.dumps(summarize_rollback(34),indent=1))
print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Light Conditions"]},separators=(",",":")))
```
