I will quantify the action effect of this edit, not just the reduction in its targeted activation. In particular, I will check whether failed-background actions move toward their matched successful controls and whether light-control actions remain largely unchanged.
```python
def load_probe(p):
    meta = json.loads((W / f"probe_{p}" / "results.json").read_text())
    meta = sorted(meta, key=lambda r: int(r["idx"]))
    a = np.stack([np.load(r["action"]) for r in meta])
    an = np.stack([np.load(r["action_norm"]) for r in meta])
    acts = {}
    if all("activations" in r for r in meta):
        zz = [np.load(r["activations"]) for r in meta]
        for key in zz[0].files:
            acts[key] = np.stack([z[key] for z in zz])
    return a, an, acts, meta

orig_a, orig_an, orig_mlp, probe_meta = load_probe(26)
failure_manifest = readj("failure_specific_manifest.json")
fp = np.array(failure_manifest["failure_pairs"], dtype=int)
good_bg = np.array(failure_manifest["successful_background_indices"], dtype=int)
failed_bg = np.unique(fp[:,0])
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))

def summarize_late_probe(p, dose, kind="target"):
    a, an, acts, meta = load_probe(p)
    assert len(a) == len(orig_a) == 91
    before = orig_a[fp[:,0]]-orig_a[fp[:,1]]
    after = a[fp[:,0]]-a[fp[:,1]]
    result = {
        "pass":p, "kind":kind, "dose":dose,
        "background_action_change_rms":rms(a[:54]-orig_a[:54]),
        "light_action_change_rms":rms(a[54:]-orig_a[54:]),
        "successful_bg_action_change_rms":rms(a[good_bg]-orig_a[good_bg]),
        "failed_bg_action_change_rms":rms(a[failed_bg]-orig_a[failed_bg]),
        "failure_control_discrepancy_rms_original":rms(before),
        "failure_control_discrepancy_ratio":rms(after)/rms(before),
        "failure_to_original_control_ratio":rms(a[fp[:,0]]-orig_a[fp[:,1]])/rms(before),
        "failure_action_change_by_channel":[rms((a-orig_a)[failed_bg,:,c]) for c in range(7)],
        "light_action_change_by_channel":[rms((a-orig_a)[54:,:,c]) for c in range(7)],
        "failure_pair_ratios":[rms(after[j])/rms(before[j]) for j in range(len(fp))]
    }
    for L in late_layers:
        k=f"mean::action.{L}.mlp_out::action"
        u=late_data[f"target{L}"]
        mu=late_data[f"mean{L}"]
        z0=(orig_mlp[k]-mu)@u
        z1=(acts[k]-mu)@u
        result[f"layer{L}_target_projection_ratio"] = rms(z1[failed_bg])/rms(z0[failed_bg])
    result["late_mlp_change_rms"] = rms(
        acts["mean::action.29.mlp_out::action"]-orig_mlp["mean::action.29.mlp_out::action"])
    (W/f"probe{p}_summary.json").write_text(json.dumps(result,indent=2))
    return result

s31=summarize_late_probe(31,.5)
print(json.dumps(s31,indent=1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass31: late-MLP half-dose causal probe\n- "+json.dumps(s31)+"\n")
print("Set preplanned full dose:", set_late_mlp_edit(1.0))
```
