Two readout directions now recur across task subsets and carry camera-pair variation while overlapping little with the controls. I will remove only those two directions from the action head, leaving its bias unchanged. This changes about 4.2% of the head’s weight norm and exactly preserves its action-encoder mapping in linear weight space.
```python
readout_camera_U=vp[:2].T.contiguous()
readout_protect=P
def set_readout_projection(alpha, tag=None, U=None):
    clear_edits()
    U = readout_camera_U if U is None else U
    name="mot.mixtures.action.head.weight"
    h0=load_original(name)
    he=h0-alpha*(h0@U)@U.T
    save_edit(name,he)
    meta={"mechanism":"Remove final-residual camera contrast orthogonal to control and action-encoder subspaces",
          "rank":U.shape[1],"alpha":alpha,"weight_relative_change":float((he-h0).norm()/h0.norm()),
          "protected_subspace_rank":int(readout_protect.shape[1]),
          "protected_linear_error_max":float(((he-h0)@readout_protect).abs().max()),
          "encoder_mapping_error_max":float(((he-h0)@encoder).abs().max())}
    if tag:
        snapshot(tag,meta)
    return meta
print(json.dumps(set_readout_projection(1.0,"readout_camera_rank2_full"),indent=2))
np.save(W/"readout_camera_U.npy",readout_camera_U.numpy())
ledger=json.loads((W/"pass_ledger.json").read_text())
ledger += [{"pass":11,"type":"rollout","edit":"original","episodes":111,
            "camera_success":"40/74","light_success":"35/37","purpose":"Broader matched discovery"},
           {"pass":12,"type":"probe","edit":"original","inputs":111,"noise_seed":46001,
            "purpose":"Post-block residuals on broader matched discovery"}]
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with notes.open("a") as f:
    f.write("\n## Passes 10-12: residual/readout diagnosis\n"
            "- Post-block camera differences accumulate across many layers; L11 and L16 contribute ~5.5% and 7.4% of final contrast, not dominant alone.\n"
            "- Broader original panel: 37 task groups, 74 camera (40 success) + 37 light (35 success). All 37 camera pairs have exactly matching wrist pixels and proprioception.\n"
            "- Combined 56 matched pairs: final camera contrast outside a protected control/action-encoder subspace (rank79) retains 72.2% of energy.\n"
            "- Leading two protected camera directions reproduce across disjoint task subsets (principal cosines .717/.597); rank2 selected to avoid unstable higher modes.\n"
            "- Candidate: action.head.weight <- W - W U U^T, U two orthonormal camera directions, orthogonal to protected span. Bias and all other weights original.\n"
            "- This is a direct spectral edit; no optimization or training. Candidate unvalidated, current best original. Passes 12/40.\n")
print("Active:",edited())
```
