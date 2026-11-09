The untouched set added two repairs and no regressions. I will now run mechanism controls on development inputs only: first, changing each target row’s magnitude without changing its direction; then preserving its original magnitude while applying the selected semantic direction. These are temporary diagnostic edits, not new candidates.
```python
final_validation=both_seed_validation+pair_roll(37)
print("All frozen non-development validation:",paired_stats(final_validation))
print("Clustered by variant:",clustered_stats(final_validation))
touched_val=[p for p in final_validation
             if set(sp.encode(p["original"]["instruction"]+"\n",add_bos=True)) & set(spec)]
print("Validation pairs containing edited tokens:",paired_stats(touched_val))
print("\nMatched diagnostic input checks:")
for label,i,c in pairs_diag[:-1]:
    a=torch.load(probe_inputs[i],map_location="cpu",weights_only=False)
    b=torch.load(probe_inputs[c],map_location="cpu",weights_only=False)
    print(label,"noise",np.array_equal(a["noise"],b["noise"]),
          "base",np.array_equal(a["element"]["observation/image"],b["element"]["observation/image"]),
          "wrist",np.array_equal(a["element"]["observation/wrist_image"],b["element"]["observation/wrist_image"]))

frozen_arr=np.load(cp_exact/(E_NAME+".npy"),mmap_mode="r")
frozen_rows=torch.from_numpy(np.array(frozen_arr[ids_edit],dtype=np.float32))
original_rows=E0[ids_edit].clone()
orig_norm=original_rows.norm(dim=1,keepdim=True)
final_norm=frozen_rows.norm(dim=1,keepdim=True)
norm_only_rows=original_rows*(final_norm/orig_norm)
direction_only_rows=frozen_rows*(orig_norm/final_norm)
norm_ratios=(final_norm/orig_norm).squeeze()
print("Target/original row norm ratio min/median/max:",
      [round(float(x),4) for x in (norm_ratios.min(),norm_ratios.median(),norm_ratios.max())])
control_dir=W/"causal_controls"
control_dir.mkdir(exist_ok=True)
np.save(control_dir/"norm_only_rows.npy",norm_only_rows.numpy())
np.save(control_dir/"direction_only_rows.npy",direction_only_rows.numpy())

def apply_trial_rows(rows,label):
    t=E0.clone()
    t[ids_edit]=rows.to(torch.bfloat16).float()
    save_edit(E_NAME,t)
    del t
    (control_dir/"current_trial.json").write_text(json.dumps({"label":label,"rows":ids_edit}))
    print("TEMPORARY CURRENT EDIT:",label)
apply_trial_rows(norm_only_rows,"norm_only")
req38=dict(json.loads((W/"mechanism_probe_request.json").read_text()))
(W/"causal_probe_request.json").write_text(json.dumps(req38,indent=1))
with (W/"notes.md").open("a") as f:
    f.write("\nPass37 untouched65 variants,128 paired trials across seeds61/67:115->117,2 fixes/0 regressions. "
            "All frozen non-development validation "+str(clustered_stats(final_validation))+".\n"
            "IMPORTANT CURRENT EDIT TEMPORARY norm_only control (after code41). "
            "Pass38 planned probe on existing16 dev inputs. Controls rows saved causal_controls/norm_only_rows.npy "
            "and direction_only_rows.npy. Final selected checkpoint remains checkpoints/lex_dest_exact; "
            "MUST RESTORE before subsequent performance audits. No final-candidate selection from these controls.\n")
print("Probe request prepared:",len(req38["inputs"]),"inputs; same as probes22/23.")
```
