I will summarize the early-group test, then narrow the edit to middle/late directions whose texture-contrast variance is at least five times their light-control variance. This avoids directions that are strongly task-dependent, including the late layer-28 directions. The narrowed candidate will then face paired rollouts rather than further probe-only selection.
```python
s12=compare_probe(12)
(W/"probe12_summary.json").write_text(json.dumps(s12,indent=2))
# Fixed analytic specificity rule, restricted to depth groups with demonstrated action influence.
selected={}
for L in range(10,30):
    ratios=np.array(projection_info[L]["direction_contrast_to_light_variance"][:2])
    inds=np.flatnonzero(ratios>=5.0).tolist()
    if inds:
        selected[L]=inds
print("Specificity-selected directions:",selected)
def apply_selected(dose=0.5, control=False):
    clear_edits()
    report=[]
    rng=np.random.default_rng(83291)
    for L,inds in selected.items():
        U=torch.from_numpy(projection_data[f"U{L}"][:,inds].copy()).float()
        mu=torch.from_numpy(projection_data[f"mean{L}"].copy()).float()
        stem=f"mot.mixtures.action.blocks.{L}.self_attn.o"
        wo=load_original(stem+".weight"); bo=load_original(stem+".bias")
        target_dw=-dose*U@(U.T@wo)
        effdose=dose
        if control:
            V=torch.from_numpy(projection_data[f"U{L}"]).float()
            R=torch.from_numpy(rng.standard_normal((1024,len(inds))).astype(np.float32))
            R=R-V@(V.T@R)
            U=torch.linalg.qr(R,mode="reduced").Q
            raw=U@(U.T@wo)
            effdose=float(target_dw.norm()/raw.norm())
        dw=-effdose*U@(U.T@wo)
        db=effdose*U@(U.T@(mu-bo))
        save_edit(stem+".weight",wo+dw)
        save_edit(stem+".bias",bo+db)
        anchored=mu-effdose*U@(U.T@(mu-bo))+db
        report.append({"layer":L,"directions":inds,"effective_dose":effdose,
                       "weight_relative_change":float(dw.norm()/wo.norm()),
                       "anchor_error":float((anchored-mu).abs().max())})
    spec={"family":"specific_centered_contrast_projection","selected":selected,
          "dose":dose,"control":control,"tensors":report,
          "selection_rule":"top2 paired contrast directions; light variance ratio >=5; layers10-29"}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
candidate=apply_selected(0.5)
print("Applied specificity-selected candidate:",
      [(x["layer"],x["directions"],round(x["weight_relative_change"],4)) for x in candidate["tensors"]])
(W/"selected_projection_directions.json").write_text(json.dumps(selected,indent=2))
# Store probe and rollout manifests for recovery, independently of candidate outcome.
probe12paths=[r["input"] for r in read_probe(12)]
(W/"probe12_paths.json").write_text(json.dumps(probe12paths))
with (W/"notes.md").open("a") as f:
    f.write(f"- Pass 12 early L0-9 rank2 dose .5: pair RMS ratio {s12['aggregate_pair_ratio']:.6f}; BG change {s12['bg_change_rms']:.6f}, light {s12['light_change_rms']:.6f}.\n"
            "- Narrowed analytic candidate: leading contrast directions with texture/light variance ratio >=5 in layers10-29, dose .5, light-mean anchored. Selection uses diagnostic probes only, no validation outcomes.\n"
            f"- Selected directions: {selected}. Passes used 12/40.\n")
```