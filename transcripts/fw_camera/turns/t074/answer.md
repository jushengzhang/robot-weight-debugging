The half-strength edit gained 7.0 points with a 3.0-point standard error, comparable to the full edit with half the weight change. I will prefer it provisionally, then test a rank- and norm-matched edit in unrelated directions to check whether the benefit depends on the diagnosed camera subspace.
```python
r16=json.loads((W/"probe_16"/"results.json").read_text())
a16=np.stack([np.load(r["action"]) for r in r16]).reshape(19,3,32,7)
s16=action_probe_summary(orig_actions,a16)
print("Half-dose probe:",json.dumps(s16,indent=2))
dhalf=(a16[:,:,:10,:3]-orig_actions[:,:,:10,:3]).reshape(-1).astype(float)
dfull=(a13[:,:,:10,:3]-orig_actions[:,:,:10,:3]).reshape(-1).astype(float)
dose_slope=float(dhalf@dfull/(dfull@dfull))
dose_cos=float(dhalf@dfull/(np.linalg.norm(dhalf)*np.linalg.norm(dfull)))
print("Action-change half/full slope",round(dose_slope,4),"cosine",round(dose_cos,4))
print("Half camera rollout:",paired_summary(17))

# Deterministic unrelated directions, orthogonal to the same protected span and to U.
gen=torch.Generator().manual_seed(9817)
Vn=torch.randn((1024,2),generator=gen)
Vn-=readout_protect@(readout_protect.T@Vn)
Vn-=readout_camera_U@(readout_camera_U.T@Vn)
Vn=torch.linalg.qr(Vn,mode="reduced").Q
target_delta=.5*(head@readout_camera_U)@readout_camera_U.T
base_control_delta=(head@Vn)@Vn.T
control_alpha=float(target_delta.norm()/base_control_delta.norm())
print("Negative-control matched alpha:",round(control_alpha,5))
ctrlmeta=set_readout_projection(control_alpha,None,Vn)
ctrlmeta.update({"mechanism":"Negative control: random rank2 directions orthogonal to protected and camera spans",
                 "seed":9817,"norm_matched_to":"readout_camera_rank2_half",
                 "camera_subspace_overlap_max":float((Vn.T@readout_camera_U).abs().max())})
snapshot("readout_rank2_negative_control",ctrlmeta)
np.save(W/"readout_negative_U.npy",Vn.numpy())
print("Control edit:",ctrlmeta)
ledger=json.loads((W/"pass_ledger.json").read_text())
ledger += [{"pass":16,"type":"probe","edit":"readout_camera_rank2_half","inputs":57,"noise_seed":46001,
            **s16,"half_full_action_slope":dose_slope,"half_full_action_cosine":dose_cos},
           {"pass":17,"type":"rollout","edit":"readout_camera_rank2_half","category":"Camera Viewpoints",
            **paired_summary(17)}]
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
with notes.open("a") as f:
    f.write("\n## Passes 16-17: fixed-subspace dose response\n"
            f"- Half rank2 projection camera-pair action RMS {s16['edited_camera_pair_rms']:.6f}; half/full action-change slope {dose_slope:.3f}, cosine {dose_cos:.3f}.\n"
            "- Half dose rollout:64/128 original vs73/128 edited; +7.03pp SE2.97 (12 gains,3 losses).\n"
            "- Prefer half dose provisionally: similar discovery benefit with 2.087% head-weight change. Not yet independently validated.\n"
            "- Negative control: deterministic random rank2 in protected complement, orthogonal to camera U; alpha chosen solely to match edit Frobenius norm.\n"
            "- Current active edit NEGATIVE CONTROL for causal test. Passes17/40.\n")
```
